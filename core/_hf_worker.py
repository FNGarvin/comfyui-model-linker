"""
Standalone worker for downloading a single file from HuggingFace via
huggingface_hub.

This file is deliberately NOT imported by the rest of this package -- it is
launched with `sys.executable <this file>` as its own OS process (see
hf_downloader.py). That's on purpose: ComfyUI loads custom node packages
through a dynamic importlib trick keyed on the node's absolute install path
(see nodes.py's load_custom_node / get_module_name, which does
`sys_module_name = module_path.replace(".", "_x_")` and injects the result
straight into sys.modules). That name isn't a real, path-discoverable
package, so a freshly spawned interpreter has no way to `import` anything
from this package by module name -- which rules out Python's own
`multiprocessing` (it pickles a reference to the target function and expects
the child to re-import its defining module). Running this file directly by
path sidesteps the problem entirely: it's executed as a normal __main__
script, using only huggingface_hub/hf_xet and the stdlib, and talks to the
parent over stdin/stdout instead.

Why this doesn't just call hf_hub_download() and read its tqdm_class hook
(the first version of this file did): confirmed empirically (see project
history) that hf_hub_download's built-in Xet integration wraps hf_xet's
*deprecated* download_files() call, whose progress callback just doesn't
fire incrementally during the transfer for at least some file sizes -- 0%,
then one jump to completion. hf_xet's newer XetSession/XetFileDownloadGroup
API exposes a `group.progress()` method that can be *polled* on our own
schedule instead of depending on how often the native code calls back into
Python, which tested as actually responsive. So: files hosted on Xet storage
go through that path directly; anything else (still on classic Git-LFS-over-
HTTP, or if the Xet path raises for any reason) falls back to the plain
hf_hub_download() call, which already has smooth progress on its own since
that's a normal sequential HTTP stream.

Protocol:
  stdin:  one JSON object, one line:
          {"repo_id": ..., "filename": ..., "revision": ...,
           "local_dir": ..., "token": <str|false>, "fallback_token": <str|null>}
  stdout: newline-delimited JSON events, each carrying "t" (time.time()):
          {"event": "progress", "n": <int>, "total": <int>}
          {"event": "retrying"}
          {"event": "done", "path": <str>}
          {"event": "error", "message": <str>}
          {"event": "log", "message": <str>, ...extra diagnostic fields}
          The "log" event is verbose diagnostic detail (elapsed times, byte
          counts, which code path was taken, poll counts) -- not needed for
          normal operation, but means a real run's console output is enough
          to reconstruct exactly what happened without re-instrumenting.
"""

import os
import sys
import json
import time
import threading


def emit(event, **fields):
    sys.stdout.write(json.dumps({"event": event, "t": time.time(), **fields}) + "\n")
    sys.stdout.flush()


def log(message, **fields):
    emit("log", message=message, **fields)


def _gated_message(status, fallback_token):
    if status in (401, 403):
        if fallback_token:
            return f"Unauthorized (HTTP {status}): the HuggingFace token was rejected."
        return (
            f"Unauthorized (HTTP {status}): this model is gated. Set HF_TOKEN or "
            "HUGGING_FACE_HUB_TOKEN in the environment ComfyUI runs in and retry."
        )
    if status == 404:
        return "Model not found (HTTP 404): the file may have been moved or deleted."
    return None


class _ProgressTqdm:
    """Minimal tqdm-compatible shim for the hf_hub_download fallback path.
    It only ever calls update()/close() on the instances it creates and
    reads .total/.n -- it doesn't need the real rendering machinery."""

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total") or 0
        self.n = 0
        self._last_emit = 0.0

    def update(self, n=1):
        self.n += n
        now = time.monotonic()
        if now - self._last_emit >= 0.2:
            self._last_emit = now
            emit("progress", n=self.n, total=self.total)

    def close(self):
        emit("progress", n=self.n, total=self.total)

    def set_description(self, *args, **kwargs):
        pass

    def refresh(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _download_via_xet(metadata, token, local_dir, filename):
    """Download a Xet-hosted file with real, pollable progress. Writes
    directly to an explicit flat path under local_dir -- unlike
    hf_hub_download's local_dir mode, start_download_file() takes whatever
    destination path we hand it, so there's no repo-structure replication
    to flatten afterward for this path."""
    import hf_xet
    from huggingface_hub.utils import build_hf_headers
    from huggingface_hub.utils._xet import refresh_xet_connection_info

    headers = build_hf_headers(token=token)

    t0 = time.time()
    conn = refresh_xet_connection_info(file_data=metadata.xet_file_data, headers=headers)
    log("xet connection info refreshed", elapsed=round(time.time() - t0, 3), endpoint=conn.endpoint)

    session = hf_xet.XetSession()
    group = session.new_file_download_group(
        endpoint=conn.endpoint,
        token=conn.access_token,
        token_expiry_unix_secs=conn.expiration_unix_epoch,
        token_refresh_url=metadata.xet_file_data.refresh_route,
        token_refresh_headers=headers,
    )

    os.makedirs(local_dir, exist_ok=True)
    dest_path = os.path.join(local_dir, filename.split("/")[-1])
    file_info = hf_xet.XetFileInfo(hash=metadata.xet_file_data.file_hash, file_size=metadata.size)

    t_start = time.time()
    group.start_download_file(file_info, dest_path)
    log("xet download started", dest_path=dest_path, total=metadata.size)

    result = {}

    def waiter():
        result["report"] = group.wait_to_finish()

    wait_thread = threading.Thread(target=waiter, daemon=True)
    wait_thread.start()

    last_completed = -1
    poll_count = 0
    while wait_thread.is_alive():
        p = group.progress()
        poll_count += 1
        if p.total_bytes_completed != last_completed:
            last_completed = p.total_bytes_completed
            total = p.total_bytes or metadata.size
            emit("progress", n=p.total_bytes_completed, total=total)
            log(
                "xet progress tick",
                elapsed=round(time.time() - t_start, 2),
                completed=p.total_bytes_completed,
                total=p.total_bytes,
                rate_bytes_per_sec=p.total_bytes_completion_rate,
                poll_count=poll_count,
            )
        time.sleep(0.3)

    wait_thread.join(timeout=10)
    log(
        "xet download finished",
        elapsed=round(time.time() - t_start, 2),
        poll_count=poll_count,
        report=str(result.get("report")),
    )

    if not os.path.exists(dest_path):
        raise RuntimeError("xet download reported completion but the destination file is missing")

    emit("progress", n=metadata.size, total=metadata.size)
    emit("done", path=dest_path)


def _download_via_hf_hub_download(repo_id, filename, revision, local_dir, token):
    """Classic path: hf_hub_download's normal sequential-HTTP download,
    which already has smooth progress on its own via the tqdm_class hook.
    Used for anything not on Xet storage, and as a fallback if the Xet path
    above raises for any reason."""
    import huggingface_hub

    t0 = time.time()
    path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
        token=token,
        tqdm_class=_ProgressTqdm,
    )
    log("hf_hub_download path finished", elapsed=round(time.time() - t0, 2))
    emit("done", path=path)


def main():
    request = json.loads(sys.stdin.readline())

    repo_id = request["repo_id"]
    filename = request["filename"]
    revision = request.get("revision") or "main"
    local_dir = request["local_dir"]
    token = request.get("token", False)
    fallback_token = request.get("fallback_token")

    try:
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        from huggingface_hub.errors import HfHubHTTPError
    except Exception as e:
        emit("error", message=f"huggingface_hub is not available in this environment: {e}")
        return

    # Resolve metadata first -- this is also our one auth checkpoint: if it
    # succeeds (anonymously or otherwise), the same token is reused for
    # whichever download path we take below, since HF enforces gating
    # consistently across the metadata and file endpoints.
    url = hf_hub_url(repo_id, filename, revision=revision)
    log("resolving metadata", url=url, token_mode=("string" if isinstance(token, str) else token))
    t0 = time.time()
    try:
        try:
            metadata = get_hf_file_metadata(url, token=token)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403) and token is False and fallback_token:
                emit("retrying")
                token = fallback_token
                metadata = get_hf_file_metadata(url, token=token)
            else:
                raise
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        emit("error", message=_gated_message(status, fallback_token) or str(e))
        return
    except Exception as e:
        emit("error", message=str(e))
        return

    log(
        "metadata resolved",
        elapsed=round(time.time() - t0, 3),
        size=metadata.size,
        xet=metadata.xet_file_data is not None,
        etag=metadata.etag,
    )

    if metadata.xet_file_data is not None:
        try:
            _download_via_xet(metadata, token, local_dir, filename)
            return
        except Exception as e:
            log("xet path raised, falling back to hf_hub_download", error=str(e))

    try:
        _download_via_hf_hub_download(repo_id, filename, revision, local_dir, token)
    except Exception as e:
        emit("error", message=str(e))


if __name__ == "__main__":
    main()
