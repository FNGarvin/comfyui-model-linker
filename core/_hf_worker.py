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

Progress design (see project history for the full trail -- this went
through several dead ends before landing here): earlier versions of this
file hand-rolled a direct hf_xet XetSession/XetFileDownloadGroup integration
to get real incremental progress, reaching into huggingface_hub's *private*
utils._xet module. That broke the first time it hit a different
huggingface_hub version in the wild (refresh_xet_connection_info doesn't
exist in 1.26.0 -- the internals were rewritten). Turns out that rewrite
also fixed the underlying problem at the source: huggingface_hub's Xet
integration now drives a *dual* progress bar (network transfer bytes,
updated continuously, vs. file-reconstruction bytes, which lags behind since
it's only counted once chunks are verified and flushed to disk) and will
call `update_transfer(n)` / `set_transfer_postfix_str(...)` on a supplied
tqdm_class *if that class defines them* -- confirmed empirically to fire
~10x/second with real incremental byte counts. So: no more private-API
reimplementation needed. Plain hf_hub_download(..., tqdm_class=...) is
enough, as long as the tqdm shim implements that (undocumented but stable
enough in practice) dual-method contract. On older huggingface_hub versions
that don't know about update_transfer, it's simply never called and this
falls back to whatever granularity the reconstruction-only update() gets --
today's existing behavior, not a regression.

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
"""

import sys
import json
import time


def emit(event, **fields):
    sys.stdout.write(json.dumps({"event": event, "t": time.time(), **fields}) + "\n")
    sys.stdout.flush()


def log(message, **fields):
    emit("log", message=message, **fields)


class _ProgressTqdm:
    """tqdm-compatible shim for hf_hub_download's tqdm_class hook.

    Implements both the standard surface (update()/close(), driven by the
    "reconstruction" bar -- bytes verified and flushed to disk, which for
    Xet-accelerated downloads can lag well behind the network) and the
    update_transfer()/set_transfer_postfix_str() pair huggingface_hub's
    dual-bar Xet progress reporter calls for actual network bytes received.
    Both report increments toward the *same* eventual total, not two
    different things to sum -- tracked separately and reported as whichever
    is further along, so this never double-counts and never regresses if a
    given huggingface_hub version only calls one of the two.
    """

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total") or 0
        self._reconstruction_n = 0
        self._transfer_n = 0
        self.n = 0
        self._last_emit = 0.0

    def _report(self, force=False):
        self.n = max(self._reconstruction_n, self._transfer_n)
        now = time.monotonic()
        if force or now - self._last_emit >= 0.2:
            self._last_emit = now
            emit("progress", n=self.n, total=self.total)

    def update(self, n=1):
        self._reconstruction_n += n
        self._report()

    def update_transfer(self, n=1):
        self._transfer_n += n
        self._report()

    def set_transfer_postfix_str(self, *args, **kwargs):
        pass

    def close(self):
        self._report(force=True)

    def set_description(self, *args, **kwargs):
        pass

    def refresh(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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


def main():
    request = json.loads(sys.stdin.readline())

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

    t0 = time.time()
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
        log("download finished", elapsed=round(time.time() - t0, 2))
        emit("done", path=path)
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        emit("error", message=_gated_message(status, fallback_token) or str(e))
    except Exception as e:
        emit("error", message=str(e))


if __name__ == "__main__":
    main()
