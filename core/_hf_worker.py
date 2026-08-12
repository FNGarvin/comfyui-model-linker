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
script, using only huggingface_hub and the stdlib, and talks to the parent
over stdin/stdout instead.

Protocol:
  stdin:  one JSON object, one line:
          {"repo_id": ..., "filename": ..., "revision": ...,
           "local_dir": ..., "token": <str|false>, "fallback_token": <str|null>}
  stdout: newline-delimited JSON events as the download progresses:
          {"event": "progress", "n": <int>, "total": <int>}
          {"event": "retrying"}
          {"event": "done", "path": <str>}
          {"event": "error", "message": <str>}
"""

import sys
import json
import time


def emit(event, **fields):
    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()


class _ProgressTqdm:
    """Minimal tqdm-compatible shim. hf_hub_download only ever calls
    update()/close() on the instances it creates and reads .total/.n --
    it doesn't need the real rendering machinery, just this surface."""

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total") or 0
        self.n = 0
        self._last_emit = 0.0

    def update(self, n=1):
        self.n += n
        now = time.monotonic()
        # Throttle -- an active 1GB/s+ transfer would otherwise flood the
        # pipe with a JSON line per chunk.
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


def main():
    line = sys.stdin.readline()
    request = json.loads(line)

    repo_id = request["repo_id"]
    filename = request["filename"]
    revision = request.get("revision") or "main"
    local_dir = request["local_dir"]
    token = request.get("token", False)
    fallback_token = request.get("fallback_token")

    try:
        import huggingface_hub
        from huggingface_hub.errors import HfHubHTTPError
    except Exception as e:
        emit("error", message=f"huggingface_hub is not available in this environment: {e}")
        return

    def attempt(use_token):
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
            token=use_token,
            tqdm_class=_ProgressTqdm,
        )

    try:
        try:
            path = attempt(token)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            # Only escalate when the first attempt was deliberately
            # anonymous (token is False, not just falsy) -- if the caller
            # already sent a token and it was refused, there's nothing
            # left to fall back to.
            if status in (401, 403) and token is False and fallback_token:
                emit("retrying")
                path = attempt(fallback_token)
            else:
                raise
        emit("done", path=path)
        return
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            if fallback_token:
                message = f"Unauthorized (HTTP {status}): the HuggingFace token was rejected."
            else:
                message = (
                    f"Unauthorized (HTTP {status}): this model is gated. Set HF_TOKEN or "
                    "HUGGING_FACE_HUB_TOKEN in the environment ComfyUI runs in and retry."
                )
        elif status == 404:
            message = "Model not found (HTTP 404): the file may have been moved or deleted."
        else:
            message = str(e)
        emit("error", message=message)
    except Exception as e:
        emit("error", message=str(e))


if __name__ == "__main__":
    main()
