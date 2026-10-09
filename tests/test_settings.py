"""
Tests for core/settings.py — run with: python tests/test_settings.py

Stubs folder_paths.get_user_directory with a temp dir; no network calls.
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER_DIR = tempfile.mkdtemp(prefix='linker_settings_')
fp = types.ModuleType('folder_paths')
fp.get_user_directory = lambda: USER_DIR
sys.modules['folder_paths'] = fp

for var in ('HF_TOKEN', 'CIVITAI_API_TOKEN'):
    os.environ.pop(var, None)

from core import settings  # noqa: E402


def test_saved_token_is_used_and_hidden():
    settings.save_settings({'hf_token': 'hf_abcdefghijklmnop'})
    assert settings.get_token('huggingface') == 'hf_abcdefghijklmnop'
    view = settings.public_view()
    assert view['huggingface'] == {'set': True, 'source': 'settings', 'env_var': 'HF_TOKEN', 'hint': '…mnop'}
    assert 'hf_abcdefghijklmnop' not in json.dumps(view)


def test_empty_string_clears_token():
    settings.save_settings({'civitai_key': 'civitai-secret-key-1234'})
    settings.save_settings({'civitai_key': ''})
    assert settings.get_token('civitai') is None
    assert settings.public_view()['civitai']['set'] is False


def test_env_var_fallback():
    settings.save_settings({'civitai_key': ''})
    os.environ['CIVITAI_API_TOKEN'] = 'env-key-0000-9999'
    try:
        assert settings.get_token('civitai') == 'env-key-0000-9999'
        assert settings.public_view()['civitai']['source'] == 'env'
    finally:
        del os.environ['CIVITAI_API_TOKEN']


def test_auth_headers_only_for_matching_host():
    settings.save_settings({'hf_token': 'hf_abcdefghijklmnop', 'civitai_key': 'civ_1234567890'})
    assert settings.auth_headers_for_url('https://huggingface.co/a/b/resolve/main/x.safetensors') == \
        {'Authorization': 'Bearer hf_abcdefghijklmnop'}
    assert settings.auth_headers_for_url('https://civitai.com/api/download/models/1') == \
        {'Authorization': 'Bearer civ_1234567890'}
    assert settings.auth_headers_for_url('https://example.com/x.safetensors') == {}


def test_settings_file_lives_in_user_dir():
    assert settings.get_settings_path() == os.path.join(USER_DIR, 'model_linker_settings.json')


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
