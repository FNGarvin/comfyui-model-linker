"""
Tests for download sources — run with: python tests/test_sources.py

Network calls are replaced with fakes; the bundled HF index is read as-is.
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fp = types.ModuleType('folder_paths')
fp.get_user_directory = lambda: tempfile.mkdtemp(prefix='linker_sources_')
fp.folder_names_and_paths = {}
sys.modules['folder_paths'] = fp

from core import source_chain  # noqa: E402
from core.linker import extract_workflow_urls  # noqa: E402
from core.sources import civarchive, hf_index  # noqa: E402
from core.sources.huggingface import build_search_queries  # noqa: E402

SHA = 'df7c757437ef3696e76ee5cc18c063681478639132547b6292b0b4d773814ba5'


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_hf_index_finds_comfy_org_split_file():
    result = hf_index.lookup_by_filename('flux1-dev-fp8.safetensors')
    assert result is not None
    assert result['repo_id'] == 'Comfy-Org/flux1-dev'  # Comfy-Org wins over Kijai's same-named file
    assert result['url'].startswith('https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/')
    assert len(result['sha256']) == 64


def test_hf_index_hash_lookup_round_trips():
    by_name = hf_index.lookup_by_filename('flux1-dev-fp8.safetensors')
    by_hash = hf_index.lookup_by_sha256(by_name['sha256'].upper())
    assert by_hash['path'] == by_name['path'] and by_hash['match_type'] == 'hash'


def test_hf_index_ignores_folder_prefix():
    assert hf_index.lookup_by_filename('FLUX\\flux1-dev-fp8.safetensors') is not None


def test_search_queries_strip_precision_suffix():
    assert build_search_queries('flux1-dev-fp8_e4m3fn.safetensors') == ['flux1-dev-fp8_e4m3fn', 'flux1-dev']
    assert build_search_queries('clip_l.safetensors') == ['clip_l']


def test_civarchive_prefers_live_huggingface_mirror():
    payload = {'model': {'name': 'Styles', 'type': 'LORA'}, 'files': [
        {'filename': 'a.safetensors', 'url': 'https://civitai.com/api/download/models/1', 'source': 'civitai'},
        {'filename': 'a.safetensors', 'url': 'https://huggingface.co/x/y/resolve/main/gone.safetensors',
         'source': 'huggingface', 'deletedAt': '2025-01-01'},
        {'filename': 'a.safetensors', 'url': 'https://huggingface.co/x/y/resolve/main/gated.safetensors',
         'source': 'huggingface', 'is_gated': True},
        {'filename': 'a.safetensors', 'url': 'https://huggingface.co/x/y/resolve/main/a.safetensors',
         'source': 'huggingface'},
    ]}
    original = civarchive.requests.get
    civarchive.requests.get = lambda url, **kw: FakeResponse(200, payload)
    civarchive._cache.clear()
    try:
        result = civarchive.lookup_by_sha256(SHA, 'a.safetensors')
    finally:
        civarchive.requests.get = original
    assert result['url'] == 'https://huggingface.co/x/y/resolve/main/a.safetensors'
    assert result['match_type'] == 'hash' and result['name'] == 'Styles'


def test_civarchive_rejects_bad_hashes_without_calling():
    calls = []
    original = civarchive.requests.get
    civarchive.requests.get = lambda url, **kw: calls.append(url)
    try:
        assert civarchive.lookup_by_sha256('not-a-hash') is None
        assert civarchive.lookup_by_sha256('0' * 64) is None
    finally:
        civarchive.requests.get = original
    assert calls == []


def test_extract_workflow_urls_captures_hashes():
    workflow = {
        'nodes': [{'type': 'LoraLoader', 'properties': {'models': [
            {'name': 'a.safetensors', 'url': 'https://example.com/a', 'directory': 'loras',
             'hash': SHA.upper(), 'hash_type': 'SHA256'}]}}],
        'extra': {'model_resolver_hashes': {'models': [
            {'node_id': 2, 'widget_index': 0, 'path': 'sub\\b.safetensors', 'sha256': SHA}]}},
    }
    urls = extract_workflow_urls(workflow)
    assert urls['a.safetensors']['sha256'] == SHA
    assert urls['b.safetensors']['sha256'] == SHA and urls['b.safetensors']['url'] == ''


def test_chain_uses_index_before_network():
    def fail(*args, **kwargs):
        raise AssertionError('network search should not run')
    original = (source_chain.search_huggingface_for_file, source_chain.search_civitai_for_file)
    source_chain.search_huggingface_for_file = fail
    source_chain.search_civitai_for_file = fail
    try:
        result = source_chain.find_download_source('flux1-krea-dev_fp8_scaled.safetensors', category='diffusion_models')
    finally:
        source_chain.search_huggingface_for_file, source_chain.search_civitai_for_file = original
    assert result['url_source'] == 'hf_index' and result['directory'] == 'diffusion_models'


def test_chain_tries_hash_before_name_search():
    original = (source_chain.search_civitai_by_hash, source_chain.search_huggingface_for_file,
                source_chain.search_civitai_for_file)
    source_chain.search_civitai_by_hash = lambda sha, api_key=None: {
        'download_url': 'https://civitai.com/api/download/models/42', 'filename': 'private.safetensors',
        'url': 'https://civitai.com/models/7', 'name': 'Private'}
    source_chain.search_huggingface_for_file = lambda *a, **k: None
    source_chain.search_civitai_for_file = lambda *a, **k: None
    try:
        result = source_chain.find_download_source('private_renamed.safetensors', category='loras', sha256=SHA)
    finally:
        (source_chain.search_civitai_by_hash, source_chain.search_huggingface_for_file,
         source_chain.search_civitai_for_file) = original
    assert result['url'] == 'https://civitai.com/api/download/models/42'
    assert result['match_type'] == 'hash'


def test_chain_prefers_workflow_url():
    result = source_chain.find_download_source('flux1-dev-fp8.safetensors',
                                               workflow_url='https://huggingface.co/a/b/resolve/main/x.safetensors',
                                               fetch_size=False)
    assert result['url_source'] == 'workflow' and result['source'] == 'huggingface'


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
