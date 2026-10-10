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
- The token comes from Model Linker's settings (core/settings.get_token, which
  also covers the HF_TOKEN environment variable). huggingface_hub's own
  token=True / token=None resolution also reads the cached `hf auth login`
  file, which this project deliberately stays out of -- a long-running shell
  login should never silently leak into a ComfyUI download. So the worker is
  always handed an explicit string or `False`, never left to resolve it on
  its own.
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
import inspect
import shutil
import subprocess
import threading
import time
import logging
import queue as queue_mod
from typing import Optional

import huggingface_hub

from .settings import get_token
from .downloader import (
    download_progress,
    download_lock,
    cancelled_downloads,
    format_bytes,
    generate_download_id,
)

logger = logging.getLogger(__name__)

# hf_hub_download only accepts tqdm_class from huggingface_hub 1.1.0. ComfyUI
# can sit on transformers 4.x, which pins huggingface_hub below 1.0; the
# ImportError makes downloader.py use the classic engine there.
if 'tqdm_class' not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
    raise ImportError(f"huggingface_hub {huggingface_hub.__version__} is too old for the HF download engine")

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


def _staging_dir(dest_dir: str, download_id: str) -> str:
    # Dot-prefixed so core/scanner.py's directory walk (which already skips
    # anything starting with '.') never sees a partially-downloaded file in
    # here and mistakes it for an installed model.
    return os.path.join(dest_dir, f'.model_linker_tmp_{download_id}')


def _remove_staging(path: str) -> None:
    # On Windows the killed worker can still hold its files for a moment, so a
    # single rmtree right after the kill can leave the folder behind.
    for _ in range(10):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.2)


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

    # An explicit string or False, never None -- see module docstring.
    token = get_token('huggingface') or False

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
        "token": token,
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
        _remove_staging(staging)
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
    error_status = None
    cancelled = False
    total_size = 0
    last_downloaded = 0
    last_speed_time = start_time
    speed = 0.0
    last_cli_log = start_time

    def _update_progress(n, total):
        # The worker reports hf_hub_download's own tqdm counters; use them as-is.
        nonlocal last_downloaded, last_speed_time, speed, last_cli_log, total_size
        if total:
            total_size = total
        now = time.time()
        dt = now - last_speed_time
        if dt > 0:
            speed = max(0.0, (n - last_downloaded) / dt)
        last_downloaded, last_speed_time = n, now
        with download_lock:
            download_progress[download_id]['downloaded'] = n
            download_progress[download_id]['speed'] = int(speed)
            if total_size:
                download_progress[download_id]['total_size'] = total_size
                download_progress[download_id]['progress'] = min(100, int(n / total_size * 100))
        if now - last_cli_log >= 5:
            last_cli_log = now
            total_str = format_bytes(total_size) if total_size else "?"
            _safe_print(f"[Model Linker] Progress: {format_bytes(n)} / {total_str} - {format_bytes(int(speed))}/s")

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
            n = msg.get("n")
            if n is not None:
                _update_progress(n, msg.get("total"))
            elif msg.get("total"):
                total_size = msg.get("total")
                with download_lock:
                    download_progress[download_id]['total_size'] = total_size
        elif event == "log":
            # Verbose diagnostic detail from the worker -- always printed
            # (not throttled), so a real run's console output is enough to
            # reconstruct exactly what happened without re-instrumenting.
            extra = {k: v for k, v in msg.items() if k not in ('event', 't', 'message')}
            extra_str = f" {extra}" if extra else ""
            _safe_print(f"[Model Linker][hf-worker] {msg.get('message', '')}{extra_str}")
        elif event == "done":
            result_path = msg.get("path")
        elif event == "error":
            error_msg = msg.get("message")
            error_status = msg.get("status")

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
        _remove_staging(staging)
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
            _remove_staging(staging)
            with download_lock:
                download_progress[download_id]['status'] = 'error'
                download_progress[download_id]['error'] = f"Downloaded but could not move into place: {e}"
            _safe_print(f"[Model Linker] ✗ Download failed: {filename_display}")
            _safe_print(f"[Model Linker] Error: could not move into place: {e}")
            return

        _remove_staging(staging)
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
    _remove_staging(staging)
    stderr_tail = ''.join(stderr_lines).strip()
    final_error = error_msg or (stderr_tail[-500:] if stderr_tail else None) or "Download failed for an unknown reason"
    with download_lock:
        download_progress[download_id]['status'] = 'error'
        download_progress[download_id]['error'] = final_error
        if error_status in (401, 403):
            download_progress[download_id]['auth_required'] = 'huggingface'
    _safe_print(f"[Model Linker] ✗ Download failed: {filename_display}")
    _safe_print(f"[Model Linker] Error: {final_error}")


def start_background_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    dest_filename: str,
    revision: str = "main",
    display_url: Optional[str] = None,
) -> str:
    """Start a HuggingFace download in a background thread. Returns a
    download_id usable with the same /model_linker/progress polling
    endpoint the legacy downloader uses."""
    download_id = generate_download_id()

    def run():
        download_hf_model(
            repo_id, filename, dest_dir, dest_filename, download_id,
            revision=revision, display_url=display_url,
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return download_id
