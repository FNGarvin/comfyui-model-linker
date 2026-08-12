"""
Tests for core/hf_downloader.py and core/_hf_worker.py — run with:
    python tests/test_hf_downloader.py

subprocess.Popen is mocked throughout, so these never actually spawn a
worker process or touch the network. The _hf_worker tests import it
directly and exercise main() in-process instead (real subprocess behavior
is out of scope for a unit test -- see the module docstrings for why
_hf_worker.py has to be launched as a separate process by real ComfyUI
rather than imported normally).

The _hf_worker tests need huggingface_hub importable; they skip themselves
(printed as SKIP, not a failure) if it isn't installed in whatever
environment runs this file.
"""

import os
import sys
import io
import json
import shutil
import tempfile
import types
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub folder_paths so core.downloader (imported transitively by
# core.hf_downloader) doesn't need a real ComfyUI install, same convention
# as test_workflow_analyzer.py.
if 'folder_paths' not in sys.modules:
    fp = types.ModuleType('folder_paths')
    fp.get_folder_paths = lambda key: []
    sys.modules['folder_paths'] = fp

from core import hf_downloader as hfd
from core.downloader import download_progress, download_lock, cancelled_downloads


class FakeStream:
    """Stand-in for a subprocess pipe: iterable line-by-line, and supports
    .readlines() the way hf_downloader's stderr drain expects."""

    def __init__(self, lines=()):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def readlines(self):
        return self._lines


class FakeProcess:
    """Stand-in for subprocess.Popen's return value."""

    def __init__(self, stdout_lines, stderr_lines=()):
        self.stdin = mock.Mock()
        self.stdout = FakeStream(stdout_lines)
        self.stderr = FakeStream(stderr_lines)
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _forget(download_id):
    with download_lock:
        download_progress.pop(download_id, None)
    cancelled_downloads.discard(download_id)


def _with_env(**env):
    """Context manager: set env vars, restore whatever was there after."""

    class _Ctx:
        def __enter__(self):
            self._saved = {k: os.environ.get(k) for k in env}
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, *exc):
            for k, v in self._saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _Ctx()


# --- env token resolution -----------------------------------------------

def test_resolve_env_token_prefers_hf_token_over_hugging_face_hub_token():
    with _with_env(HF_TOKEN='tok-a', HUGGING_FACE_HUB_TOKEN='tok-b'):
        assert hfd._resolve_env_token() == 'tok-a'


def test_resolve_env_token_falls_back_to_hugging_face_hub_token():
    with _with_env(HF_TOKEN=None, HUGGING_FACE_HUB_TOKEN='tok-b'):
        assert hfd._resolve_env_token() == 'tok-b'


def test_resolve_env_token_none_when_unset_or_blank():
    with _with_env(HF_TOKEN=None, HUGGING_FACE_HUB_TOKEN=None):
        assert hfd._resolve_env_token() is None
    with _with_env(HF_TOKEN='   ', HUGGING_FACE_HUB_TOKEN=None):
        assert hfd._resolve_env_token() is None


# --- orchestrator: staging / move / cleanup ------------------------------

def test_happy_path_moves_file_flat_and_removes_staging_dir():
    tmp = tempfile.mkdtemp(prefix='hfdl_test_')
    download_id = 'happytest'
    try:
        dest_dir = os.path.join(tmp, 'checkpoints')
        os.makedirs(dest_dir, exist_ok=True)
        staging = hfd._staging_dir(dest_dir, download_id)

        def fake_popen(*args, **kwargs):
            # Simulate what the real worker would have produced by now:
            # the repo's nested layout replicated under the staging dir.
            nested = os.path.join(staging, 'text_encoder')
            os.makedirs(nested, exist_ok=True)
            staged_file = os.path.join(nested, 'model.safetensors')
            with open(staged_file, 'wb') as f:
                f.write(b'x' * 100)
            lines = [
                json.dumps({"event": "progress", "n": 50, "total": 100}) + "\n",
                json.dumps({"event": "progress", "n": 100, "total": 100}) + "\n",
                json.dumps({"event": "done", "path": staged_file}) + "\n",
            ]
            return FakeProcess(lines)

        with mock.patch.object(hfd.subprocess, 'Popen', side_effect=fake_popen):
            hfd.download_hf_model(
                repo_id='someuser/somerepo',
                filename='text_encoder/model.safetensors',
                dest_dir=dest_dir,
                dest_filename='model.safetensors',  # flattened, no subfolder
                download_id=download_id,
            )

        dest_path = os.path.join(dest_dir, 'model.safetensors')
        assert os.path.exists(dest_path), 'file should land flat at dest_dir, not nested'
        assert not os.path.exists(staging), 'staging dir (incl. its .cache/huggingface) must be gone'

        with download_lock:
            info = dict(download_progress.get(download_id) or {})
        assert info.get('status') == 'completed'
        assert info.get('progress') == 100
    finally:
        _forget(download_id)
        shutil.rmtree(tmp, ignore_errors=True)


def test_error_event_marks_progress_error_and_cleans_staging():
    tmp = tempfile.mkdtemp(prefix='hfdl_test_')
    download_id = 'errortest'
    try:
        dest_dir = os.path.join(tmp, 'checkpoints')
        os.makedirs(dest_dir, exist_ok=True)
        staging = hfd._staging_dir(dest_dir, download_id)

        def fake_popen(*args, **kwargs):
            os.makedirs(staging, exist_ok=True)  # worker still creates it before failing
            lines = [json.dumps({"event": "error", "message": "gated, no token available"}) + "\n"]
            return FakeProcess(lines)

        with mock.patch.object(hfd.subprocess, 'Popen', side_effect=fake_popen):
            hfd.download_hf_model(
                repo_id='someuser/gatedrepo',
                filename='model.safetensors',
                dest_dir=dest_dir,
                dest_filename='model.safetensors',
                download_id=download_id,
            )

        assert not os.path.exists(staging)
        with download_lock:
            info = dict(download_progress.get(download_id) or {})
        assert info.get('status') == 'error'
        assert 'gated' in (info.get('error') or '')
    finally:
        _forget(download_id)
        shutil.rmtree(tmp, ignore_errors=True)


def test_cancellation_kills_process_and_cleans_staging():
    tmp = tempfile.mkdtemp(prefix='hfdl_test_')
    download_id = 'canceltest'
    try:
        dest_dir = os.path.join(tmp, 'checkpoints')
        os.makedirs(dest_dir, exist_ok=True)
        staging = hfd._staging_dir(dest_dir, download_id)

        captured = {}

        def fake_popen(*args, **kwargs):
            os.makedirs(staging, exist_ok=True)
            # Mark cancelled "mid-transfer" -- before the loop consumes any
            # progress lines -- to simulate a cancel click during download.
            cancelled_downloads.add(download_id)
            proc = FakeProcess([json.dumps({"event": "progress", "n": 1, "total": 100}) + "\n"])
            captured['proc'] = proc
            return proc

        with mock.patch.object(hfd.subprocess, 'Popen', side_effect=fake_popen):
            hfd.download_hf_model(
                repo_id='someuser/somerepo',
                filename='model.safetensors',
                dest_dir=dest_dir,
                dest_filename='model.safetensors',
                download_id=download_id,
            )

        assert not os.path.exists(staging)
        assert captured['proc'].terminated
        with download_lock:
            info = dict(download_progress.get(download_id) or {})
        assert info.get('status') == 'cancelled'
        assert download_id not in cancelled_downloads, 'cancel flag should be consumed, not linger'
    finally:
        _forget(download_id)
        shutil.rmtree(tmp, ignore_errors=True)


# --- dispatch: downloader.start_background_download routes HF vs. legacy --

def test_dispatch_routes_huggingface_url_to_hf_engine():
    from core import downloader as dl

    tmp = tempfile.mkdtemp(prefix='dispatch_test_')
    try:
        with mock.patch.object(dl, 'get_download_directory', return_value=tmp), \
             mock.patch('core.hf_downloader.start_background_hf_download', return_value='fake-id') as mocked:
            result = dl.start_background_download(
                url='https://huggingface.co/someuser/somerepo/resolve/main/model.safetensors',
                filename='model.safetensors',
                category='checkpoints',
            )
        assert result == 'fake-id'
        assert mocked.called
        _, kwargs = mocked.call_args
        assert kwargs['repo_id'] == 'someuser/somerepo'
        assert kwargs['filename'] == 'model.safetensors'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dispatch_leaves_civitai_url_on_legacy_path():
    from core import downloader as dl

    tmp = tempfile.mkdtemp(prefix='dispatch_test_')
    try:
        with mock.patch.object(dl, 'get_download_directory', return_value=tmp), \
             mock.patch('core.hf_downloader.start_background_hf_download') as mocked, \
             mock.patch.object(dl, 'download_model') as mocked_legacy:
            download_id = dl.start_background_download(
                url='https://civitai.com/api/download/models/12345',
                filename='model.safetensors',
                category='checkpoints',
            )
            # start_background_download spawns download_model on a thread;
            # give it a beat to run.
            import time
            for _ in range(50):
                if mocked_legacy.called:
                    break
                time.sleep(0.02)
        assert not mocked.called, 'CivitAI URLs must never reach the HF engine'
        assert mocked_legacy.called
        _forget(download_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- _hf_worker: the anonymous-first / escalate-only-when-mandatory logic -
#
# Wraps hf_hub_download() itself -- the first (and only) network call
# main() makes. No separate metadata pre-fetch or private hf_xet API: see
# _hf_worker.py's module docstring for why the earlier custom XetSession
# reimplementation got retired (broke on a different huggingface_hub version
# in the wild) in favor of hf_hub_download's own tqdm_class hook, which
# turns out to give real incremental progress once the shim implements its
# (undocumented but stable) update_transfer contract -- covered separately
# below in the _ProgressTqdm tests.

def _fake_response(status_code):
    """HfHubHTTPError.__init__ reads response.headers.get(...) and
    response.request, not just .status_code -- a bare SimpleNamespace(
    status_code=...) blows up inside the exception's own constructor.
    Confirmed against the installed huggingface_hub's actual source rather
    than assumed."""
    return SimpleNamespace(status_code=status_code, headers={}, request=None)


def _run_worker_main(request, hf_hub_download_mock):
    """Runs core._hf_worker.main() in-process with stdin/stdout swapped for
    StringIO buffers and huggingface_hub.hf_hub_download mocked out.
    Returns the list of parsed JSON events it emitted."""
    from core import _hf_worker as worker

    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(json.dumps(request) + "\n")
    sys.stdout = io.StringIO()
    try:
        with mock.patch('huggingface_hub.hf_hub_download', hf_hub_download_mock):
            worker.main()
        output = sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout

    return [json.loads(line) for line in output.splitlines() if line.strip()]


def _skip_without_huggingface_hub(test_name):
    try:
        import huggingface_hub  # noqa: F401
        from huggingface_hub.errors import HfHubHTTPError  # noqa: F401
        return False
    except ImportError:
        print(f'SKIP {test_name} (huggingface_hub not installed)')
        return True


def test_worker_escalates_once_on_gated_response_when_fallback_available():
    if _skip_without_huggingface_hub('test_worker_escalates_once_on_gated_response_when_fallback_available'):
        return
    from huggingface_hub.errors import HfHubHTTPError

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs['token'])
        if kwargs['token'] is False:
            raise HfHubHTTPError('gated', response=_fake_response(403))
        return '/staging/model.safetensors'

    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": False, "fallback_token": "env-token-value"},
        fake_download,
    )

    assert calls == [False, 'env-token-value'], 'must try anonymous first, only escalate after a 403'
    kinds = [e['event'] for e in events]
    assert 'retrying' in kinds
    assert events[-1]["event"] == "done" and events[-1]["path"] == "/staging/model.safetensors"


def test_worker_does_not_retry_without_a_fallback_token():
    if _skip_without_huggingface_hub('test_worker_does_not_retry_without_a_fallback_token'):
        return
    from huggingface_hub.errors import HfHubHTTPError

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs['token'])
        raise HfHubHTTPError('gated', response=_fake_response(403))

    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": False, "fallback_token": None},
        fake_download,
    )

    assert calls == [False], 'no token to fall back to -- must not retry'
    assert events[-1]['event'] == 'error'
    assert 'gated' in events[-1]['message'].lower()


def test_worker_never_escalates_when_a_token_was_already_sent():
    if _skip_without_huggingface_hub('test_worker_never_escalates_when_a_token_was_already_sent'):
        return
    from huggingface_hub.errors import HfHubHTTPError

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs['token'])
        raise HfHubHTTPError('unauthorized', response=_fake_response(401))

    # "Always send" checkbox path: token is already a real string on the
    # first attempt, so there is nothing left to escalate to even though a
    # fallback_token happens to be set -- a rejected token must not retry.
    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": "already-sent-token", "fallback_token": "env-token-value"},
        fake_download,
    )

    assert calls == ['already-sent-token']
    assert events[-1]['event'] == 'error'
    assert 'rejected' in events[-1]['message'].lower()


# --- _ProgressTqdm: the reconstruction/transfer dual-signal shim ----------
#
# huggingface_hub's Xet integration drives two different progress signals on
# whatever tqdm_class it's given -- update() for "reconstruction" bytes
# (verified + flushed to disk, can lag well behind) and, if the class
# defines it, update_transfer() for raw network bytes (updates continuously,
# confirmed empirically to fire ~10x/second). Both count toward the same
# eventual total, not two different things to add together.

def test_progress_tqdm_uses_whichever_signal_is_further_along():
    from core._hf_worker import _ProgressTqdm

    bar = _ProgressTqdm(total=1000)
    bar.update_transfer(300)   # network races ahead
    assert bar.n == 300
    bar.update(100)            # reconstruction still behind -- must not win
    assert bar.n == 300
    bar.update(250)            # reconstruction catches up and passes
    assert bar.n == 350
    bar.update_transfer(700)   # network finishes
    assert bar.n == 1000


def test_progress_tqdm_never_double_counts_both_signals():
    from core._hf_worker import _ProgressTqdm

    bar = _ProgressTqdm(total=1000)
    for _ in range(10):
        bar.update_transfer(100)  # network reaches the full total...
    for _ in range(10):
        bar.update(100)           # ...then reconstruction catches up to the same total
    assert bar.n == 1000, 'reconstruction and transfer both approach the same total -- must not sum to 2000'


def test_progress_tqdm_works_when_update_transfer_is_never_called():
    # Older huggingface_hub versions that don't know about the dual-bar
    # contract simply never call update_transfer -- must not regress.
    from core._hf_worker import _ProgressTqdm

    bar = _ProgressTqdm(total=1000)
    bar.update(400)
    bar.update(600)
    assert bar.n == 1000


def test_worker_download_uses_dual_signal_tqdm_class():
    if _skip_without_huggingface_hub('test_worker_download_uses_dual_signal_tqdm_class'):
        return

    captured = {}

    def fake_download(**kwargs):
        tqdm_cls = kwargs['tqdm_class']
        bar = tqdm_cls(total=1000)
        bar.update_transfer(500)  # exercised here to prove the class hf_hub_download receives supports it
        captured['n_after_transfer_update'] = bar.n
        bar.close()
        return '/staging/model.safetensors'

    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": False, "fallback_token": None},
        fake_download,
    )

    assert captured['n_after_transfer_update'] == 500
    progress_events = [e for e in events if e['event'] == 'progress']
    assert any(e['n'] == 500 for e in progress_events), 'the transfer-driven progress must reach the parent'


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as e:
                failures += 1
                print(f'FAIL {name}: {e}')
    sys.exit(1 if failures else 0)
