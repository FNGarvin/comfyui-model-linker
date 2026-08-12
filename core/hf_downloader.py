"""
HuggingFace Native Downloader

Fetches files that live on huggingface.co via huggingface_hub instead of the
hand-rolled single-stream loop in downloader.py. That loop is untouched and
still handles CivitAI and any other direct URL -- this module only ever
handles huggingface.co.

Why this exists (see project history for the full discussion):
- huggingface_hub is already a hard transitive dependency of ComfyUI itself
  (via transformers), and on common platforms it pulls in hf_xet, which does
  genuine concurrent range-get downloading. A single requests.get() stream
  can't approach that no matter how it's tuned, so this is a real engine
  swap, not a tuning knob.

Design constraints this module exists to satisfy:
- Anonymous by default. A token is only ever sent when the user opts in via
  the "always send" checkbox, or an anonymous attempt comes back gated and a
  fallback token is available in the environment.
- The only implicit token source is HF_TOKEN / HUGGING_FACE_HUB_TOKEN read
  from os.environ by this code. huggingface_hub's own token=True / token=None
  resolution also reads the cached `hf auth login` file, which this project
  deliberately stays out of -- a long-running shell login should never
  silently leak into a ComfyUI download. So the worker is always handed an
  explicit string or `False`, never left to resolve it on its own.
- Cancellation is a real OS-level process kill, not a cooperative flag check
  -- hf_hub_download owns its read loop internally and offers no hook to
  interrupt it politely mid-transfer the way the hand-rolled loop does. The
  actual download runs in a separate worker process (core/_hf_worker.py,
  launched via subprocess rather than multiprocessing -- see that file's
  docstring for why) specifically so "cancel" can be `proc.kill()`.
- Files land in a private, dot-prefixed staging directory *inside* the real
  destination directory, then get moved to the flat path ComfyUI expects and
  the whole staging directory is discarded. This sidesteps two problems at
  once: hf_hub_download's `local_dir` mode replicates the repo's internal
  folder structure (so a file at "text_encoder/model.safetensors" in the repo
  would otherwise land nested under the destination instead of flat in it),
  and it avoids ever writing hf_hub_download's own `.cache/huggingface/`
  bookkeeping folder into the real, shared, user-visible model directory,
  where some other download could plausibly be using the same folder at the
  same time. Staging *inside* dest_dir (not a system temp directory) keeps
  the final move same-filesystem and atomic regardless of how the user has
  their model folders spread across drives.
"""

import os
import sys
import json
import shutil
import subprocess
import threading
import time
import logging
import queue as queue_mod
from typing import Optional

from .downloader import (
    download_progress,
    download_lock,
    cancelled_downloads,
    format_bytes,
    generate_download_id,
)

logger = logging.getLogger(__name__)

ENV_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_hf_worker.py")

# How long to wait for the worker process to exit on its own after we've
# stopped reading its stdout (EOF) or decided to cancel it.
_PROCESS_JOIN_TIMEOUT = 5


def _safe_print(message: str) -> None:
    """print(), but tolerant of consoles whose active codepage can't
    represent the glyphs below (e.g. Windows cmd.exe on cp1252) -- falls
    back to an ASCII-safe rendering instead of crashing the download."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(message.encode(encoding, errors='replace').decode(encoding))


def _resolve_env_token() -> Optional[str]:
    """The only implicit auth source this module uses. Deliberately not
    huggingface_hub's own token resolution, which also reads the cached
    `hf auth login` file -- see module docstring."""
    for var in ENV_TOKEN_VARS:
        val = os.environ.get(var, '').strip()
        if val:
            return val
    return None


def _staging_dir(dest_dir: str, download_id: str) -> str:
    # Dot-prefixed so core/scanner.py's directory walk (which already skips
    # anything starting with '.') never sees a partially-downloaded file in
    # here and mistakes it for an installed model.
    return os.path.join(dest_dir, f'.model_linker_tmp_{download_id}')


def _pipe_reader(pipe, out_queue):
    """Runs in its own thread; forwards lines from a subprocess pipe into a
    plain thread-safe queue.Queue so the caller can poll with a timeout
    (and therefore notice a cancellation request) instead of blocking on
    the pipe indefinitely."""
    try:
        for line in pipe:
            out_queue.put(line)
    except Exception:
        pass
    finally:
        out_queue.put(None)  # sentinel: pipe closed


def download_hf_model(
    repo_id: str,
    filename: str,
    dest_dir: str,
    dest_filename: str,
    download_id: str,
    revision: str = "main",
    send_token: bool = False,
    display_url: Optional[str] = None,
) -> None:
    """Download one file out of a HuggingFace repo straight into dest_dir,
    flattened to dest_filename. Mutates the shared download_progress dict
    the same way downloader.download_file does, so the existing polling
    endpoint and frontend work completely unchanged.
    """
    dest_path = os.path.join(dest_dir, dest_filename)
    staging = _staging_dir(dest_dir, download_id)
    start_time = time.time()

    with download_lock:
        download_progress[download_id] = {
            'status': 'starting',
            'progress': 0,
            'total_size': 0,
            'downloaded': 0,
            'filename': dest_filename,
            'url': display_url or f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}",
            'error': None,
            'speed': 0,
            'start_time': start_time,
        }

    env_token = _resolve_env_token()
    # Checkbox on + a token available -> send it from the very first
    # request. Otherwise start anonymous; the worker escalates on its own
    # only if it hits a gated/401/403 response and a fallback exists.
    initial_token = env_token if (send_token and env_token) else False
    fallback_token = None if send_token else env_token

    try:
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(staging, exist_ok=True)
    except Exception as e:
        with download_lock:
            download_progress[download_id]['status'] = 'error'
            download_progress[download_id]['error'] = f"Could not create download directory: {e}"
        return

    filename_display = os.path.basename(dest_filename)
    _safe_print(f"\n[Model Linker] Starting download: {filename_display}")
    _safe_print(f"[Model Linker] Source: HuggingFace ({repo_id})")

    request = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "local_dir": staging,
        "token": initial_token,
        "fallback_token": fallback_token,
    }

    try:
        proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        shutil.rmtree(staging, ignore_errors=True)
        with download_lock:
            download_progress[download_id]['status'] = 'error'
            download_progress[download_id]['error'] = f"Could not start download worker: {e}"
        _safe_print(f"[Model Linker] ✗ Download failed: {filename_display}")
        _safe_print(f"[Model Linker] Error: Could not start download worker: {e}")
        return

    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        proc.stdin.close()
    except Exception:
        pass  # worker will surface its own error over stdout if this mattered

    with download_lock:
        download_progress[download_id]['status'] = 'downloading'

    stdout_queue: queue_mod.Queue = queue_mod.Queue()
    stderr_lines = []
    reader = threading.Thread(target=_pipe_reader, args=(proc.stdout, stdout_queue), daemon=True)
    reader.start()
    stderr_reader = threading.Thread(
        target=lambda: stderr_lines.extend(proc.stderr.readlines() if proc.stderr else []),
        daemon=True,
    )
    stderr_reader.start()

    result_path = None
    error_msg = None
    cancelled = False
    last_downloaded = 0
    last_speed_time = start_time
    speed = 0.0
    last_cli_log = start_time

    while True:
        if download_id in cancelled_downloads:
            cancelled = True
            break
        try:
            line = stdout_queue.get(timeout=0.5)
        except queue_mod.Empty:
            continue
        if line is None:
            break  # pipe closed, worker is done producing output
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue

        event = msg.get("event")
        if event == "progress":
            n, total = msg.get("n", 0), msg.get("total", 0)
            now = time.time()
            dt = now - last_speed_time
            if dt > 0:
                speed = (n - last_downloaded) / dt
            last_downloaded, last_speed_time = n, now
            with download_lock:
                download_progress[download_id]['downloaded'] = n
                download_progress[download_id]['total_size'] = total
                download_progress[download_id]['speed'] = int(speed)
                if total:
                    download_progress[download_id]['progress'] = int(n / total * 100)
            if now - last_cli_log >= 5:
                last_cli_log = now
                total_str = format_bytes(total) if total else "?"
                _safe_print(f"[Model Linker] Progress: {format_bytes(n)} / {total_str} - {format_bytes(int(speed))}/s")
        elif event == "retrying":
            _safe_print(f"[Model Linker] {filename_display} is gated -- retrying with token from environment")
        elif event == "done":
            result_path = msg.get("path")
        elif event == "error":
            error_msg = msg.get("message")

    if download_id in cancelled_downloads:
        cancelled = True

    if cancelled:
        proc.terminate()
        try:
            proc.wait(timeout=_PROCESS_JOIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=_PROCESS_JOIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        with download_lock:
            download_progress[download_id]['status'] = 'cancelled'
        _safe_print(f"[Model Linker] Cancelled: {filename_display} - incomplete file deleted")
        cancelled_downloads.discard(download_id)
        return

    try:
        proc.wait(timeout=_PROCESS_JOIN_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()

    if result_path and os.path.exists(result_path):
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            os.replace(result_path, dest_path)
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            with download_lock:
                download_progress[download_id]['status'] = 'error'
                download_progress[download_id]['error'] = f"Downloaded but could not move into place: {e}"
            _safe_print(f"[Model Linker] ✗ Download failed: {filename_display}")
            _safe_print(f"[Model Linker] Error: could not move into place: {e}")
            return

        shutil.rmtree(staging, ignore_errors=True)
        size = os.path.getsize(dest_path)
        elapsed = time.time() - start_time
        avg_speed = size / elapsed if elapsed > 0 else 0
        with download_lock:
            download_progress[download_id]['status'] = 'completed'
            download_progress[download_id]['progress'] = 100
            download_progress[download_id]['speed'] = 0
        _safe_print(f"[Model Linker] ✓ Download complete: {filename_display}")
        _safe_print(f"[Model Linker] Size: {format_bytes(size)}, Time: {elapsed:.1f}s, Avg speed: {format_bytes(int(avg_speed))}/s")
        return

    # Error path
    shutil.rmtree(staging, ignore_errors=True)
    stderr_tail = ''.join(stderr_lines).strip()
    final_error = error_msg or (stderr_tail[-500:] if stderr_tail else None) or "Download failed for an unknown reason"
    with download_lock:
        download_progress[download_id]['status'] = 'error'
        download_progress[download_id]['error'] = final_error
    _safe_print(f"[Model Linker] ✗ Download failed: {filename_display}")
    _safe_print(f"[Model Linker] Error: {final_error}")


def start_background_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    dest_filename: str,
    revision: str = "main",
    send_token: bool = False,
    display_url: Optional[str] = None,
) -> str:
    """Start a HuggingFace download in a background thread. Returns a
    download_id usable with the same /model_linker/progress polling
    endpoint the legacy downloader uses."""
    download_id = generate_download_id()

    def run():
        download_hf_model(
            repo_id, filename, dest_dir, dest_filename, download_id,
            revision=revision, send_token=send_token, display_url=display_url,
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return download_id
