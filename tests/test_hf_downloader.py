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
import contextlib
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
# The auth checkpoint lives around get_hf_file_metadata() (the first network
# call main() makes), not around hf_hub_download() -- the same resolved
# token is then reused for whichever download path (Xet or classic) gets
# taken afterward, since HF enforces gating consistently across both
# endpoints. See _hf_worker.py's module docstring for the full picture.

def _fake_response(status_code):
    """HfHubHTTPError.__init__ reads response.headers.get(...) and
    response.request, not just .status_code -- a bare SimpleNamespace(
    status_code=...) blows up inside the exception's own constructor.
    Confirmed against the installed huggingface_hub's actual source rather
    than assumed."""
    return SimpleNamespace(status_code=status_code, headers={}, request=None)


def _fake_metadata(size=100, xet_file_data=None, etag='abc123'):
    return SimpleNamespace(size=size, etag=etag, xet_file_data=xet_file_data)


def _run_worker_main(request, get_metadata_mock, hf_hub_download_mock=None):
    """Runs core._hf_worker.main() in-process with stdin/stdout swapped for
    StringIO buffers and huggingface_hub's network-calling functions mocked
    out. Returns the list of parsed JSON events it emitted."""
    from core import _hf_worker as worker

    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(json.dumps(request) + "\n")
    sys.stdout = io.StringIO()
    try:
        patches = [mock.patch('huggingface_hub.get_hf_file_metadata', get_metadata_mock)]
        if hf_hub_download_mock is not None:
            patches.append(mock.patch('huggingface_hub.hf_hub_download', hf_hub_download_mock))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
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


def test_worker_escalates_metadata_fetch_once_on_gated_response_when_fallback_available():
    if _skip_without_huggingface_hub('test_worker_escalates_metadata_fetch_once_on_gated_response_when_fallback_available'):
        return
    from huggingface_hub.errors import HfHubHTTPError

    calls = []

    def fake_get_metadata(url, token=None, **kwargs):
        calls.append(token)
        if token is False:
            raise HfHubHTTPError('gated', response=_fake_response(403))
        return _fake_metadata()  # xet_file_data=None -> classic path

    def fake_download(**kwargs):
        assert kwargs['token'] == 'env-token-value', 'the escalated token must carry through to the download itself'
        return '/staging/model.safetensors'

    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": False, "fallback_token": "env-token-value"},
        fake_get_metadata, fake_download,
    )

    assert calls == [False, 'env-token-value'], 'must try anonymous first, only escalate after a 403'
    kinds = [e['event'] for e in events]
    assert 'retrying' in kinds
    assert events[-1]["event"] == "done" and events[-1]["path"] == "/staging/model.safetensors"


def test_worker_does_not_retry_metadata_without_a_fallback_token():
    if _skip_without_huggingface_hub('test_worker_does_not_retry_metadata_without_a_fallback_token'):
        return
    from huggingface_hub.errors import HfHubHTTPError

    calls = []

    def fake_get_metadata(url, token=None, **kwargs):
        calls.append(token)
        raise HfHubHTTPError('gated', response=_fake_response(403))

    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": False, "fallback_token": None},
        fake_get_metadata,
    )

    assert calls == [False], 'no token to fall back to -- must not retry'
    assert events[-1]['event'] == 'error'
    assert 'gated' in events[-1]['message'].lower()


def test_worker_never_escalates_metadata_when_a_token_was_already_sent():
    if _skip_without_huggingface_hub('test_worker_never_escalates_metadata_when_a_token_was_already_sent'):
        return
    from huggingface_hub.errors import HfHubHTTPError

    calls = []

    def fake_get_metadata(url, token=None, **kwargs):
        calls.append(token)
        raise HfHubHTTPError('unauthorized', response=_fake_response(401))

    # "Always send" checkbox path: token is already a real string on the
    # first attempt, so there is nothing left to escalate to even though a
    # fallback_token happens to be set -- a rejected token must not retry.
    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
         "token": "already-sent-token", "fallback_token": "env-token-value"},
        fake_get_metadata,
    )

    assert calls == ['already-sent-token']
    assert events[-1]['event'] == 'error'
    assert 'rejected' in events[-1]['message'].lower()


def test_worker_non_xet_file_uses_hf_hub_download():
    if _skip_without_huggingface_hub('test_worker_non_xet_file_uses_hf_hub_download'):
        return

    def fake_get_metadata(url, token=None, **kwargs):
        return _fake_metadata(xet_file_data=None)  # not Xet-hosted

    def fake_download(**kwargs):
        assert kwargs['token'] is False
        return '/staging/plain.safetensors'

    events = _run_worker_main(
        {"repo_id": "u/r", "filename": "plain.safetensors", "local_dir": "/staging",
         "token": False, "fallback_token": None},
        fake_get_metadata, fake_download,
    )

    assert events[-1]["event"] == "done" and events[-1]["path"] == "/staging/plain.safetensors"


def test_worker_xet_path_downloads_and_reports_real_progress():
    if _skip_without_huggingface_hub('test_worker_xet_path_downloads_and_reports_real_progress'):
        return

    tmp = tempfile.mkdtemp(prefix='xet_worker_test_')
    try:
        xet_data = SimpleNamespace(file_hash='deadbeef', refresh_route='https://example.com/refresh')

        def fake_get_metadata(url, token=None, **kwargs):
            return _fake_metadata(size=1000, xet_file_data=xet_data)

        class _FakeGroup:
            """Mirrors the real hf_xet behavior confirmed empirically:
            nothing lands on disk until wait_to_finish() completes, and
            progress() returns an increasing sequence across calls."""

            def __init__(self):
                self._sequence = [0, 400, 1000]
                self._i = 0
                self._dest = None

            def start_download_file(self, file_info, dest_path):
                self._dest = dest_path

            def progress(self):
                completed = self._sequence[min(self._i, len(self._sequence) - 1)]
                self._i += 1
                return SimpleNamespace(total_bytes_completed=completed, total_bytes=1000,
                                        total_bytes_completion_rate=500.0)

            def wait_to_finish(self):
                import time as _time
                _time.sleep(0.35)  # give the 0.3s poll loop at least one real tick
                with open(self._dest, 'wb') as f:
                    f.write(b'x' * 1000)
                return SimpleNamespace(files=1)

        fake_group = _FakeGroup()
        fake_session = mock.Mock()
        fake_session.new_file_download_group.return_value = fake_group
        fake_hf_xet = mock.Mock()
        fake_hf_xet.XetSession.return_value = fake_session
        fake_hf_xet.XetFileInfo = lambda hash, file_size=None: SimpleNamespace(hash=hash, file_size=file_size)

        saved_hf_xet = sys.modules.get('hf_xet')
        sys.modules['hf_xet'] = fake_hf_xet
        try:
            with mock.patch('huggingface_hub.utils._xet.refresh_xet_connection_info',
                             return_value=SimpleNamespace(endpoint='https://cas.example.com',
                                                           access_token='tok', expiration_unix_epoch=0)):
                events = _run_worker_main(
                    {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": tmp,
                     "token": False, "fallback_token": None},
                    fake_get_metadata,
                )
        finally:
            if saved_hf_xet is not None:
                sys.modules['hf_xet'] = saved_hf_xet
            else:
                sys.modules.pop('hf_xet', None)

        kinds = [e['event'] for e in events]
        assert 'done' in kinds
        progress_events = [e for e in events if e['event'] == 'progress']
        assert any(e['n'] > 0 for e in progress_events), 'must report real, nonzero progress before completion'
        done = [e for e in events if e['event'] == 'done'][0]
        assert os.path.exists(done['path'])
        assert os.path.getsize(done['path']) == 1000
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_worker_xet_path_failure_falls_back_to_hf_hub_download():
    if _skip_without_huggingface_hub('test_worker_xet_path_failure_falls_back_to_hf_hub_download'):
        return

    xet_data = SimpleNamespace(file_hash='deadbeef', refresh_route='https://example.com/refresh')

    def fake_get_metadata(url, token=None, **kwargs):
        return _fake_metadata(size=1000, xet_file_data=xet_data)

    def fake_download(**kwargs):
        return '/staging/fallback.safetensors'

    with mock.patch('huggingface_hub.utils._xet.refresh_xet_connection_info',
                     side_effect=RuntimeError('xet connection refused')):
        events = _run_worker_main(
            {"repo_id": "u/r", "filename": "model.safetensors", "local_dir": "/staging",
             "token": False, "fallback_token": None},
            fake_get_metadata, fake_download,
        )

    assert events[-1]["event"] == "done" and events[-1]["path"] == "/staging/fallback.safetensors"
    assert any(e['event'] == 'log' and 'falling back' in e.get('message', '') for e in events)


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
