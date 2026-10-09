"""
Tests for core/exporter.py — run with: python tests/test_exporter.py
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = tempfile.mkdtemp(prefix='linker_export_models_')
OUTSIDE_DIR = tempfile.mkdtemp(prefix='linker_export_outside_')
os.makedirs(os.path.join(MODELS_DIR, 'loras'))
INSIDE_FILE = os.path.join(MODELS_DIR, 'loras', 'a.safetensors')
OUTSIDE_FILE = os.path.join(OUTSIDE_DIR, 'secret.txt')
for path in (INSIDE_FILE, OUTSIDE_FILE):
    with open(path, 'wb') as f:
        f.write(b'hello')

fp = types.ModuleType('folder_paths')
fp.folder_names_and_paths = {'loras': ([os.path.join(MODELS_DIR, 'loras')], set())}
sys.modules['folder_paths'] = fp

from core.exporter import (  # noqa: E402
    build_model_links_markdown, compute_sha256, is_inside_model_dirs, lookup_links, strip_secrets,
)


def test_strip_secrets_removes_tokens_only():
    assert strip_secrets('https://civitai.com/api/download/models/1?token=abc&type=Model') == \
        'https://civitai.com/api/download/models/1?type=Model'
    assert strip_secrets('https://huggingface.co/a/b/resolve/main/x.safetensors') == \
        'https://huggingface.co/a/b/resolve/main/x.safetensors'
    assert strip_secrets('https://x.com/f?KEY=1&api_key=2') == 'https://x.com/f'


def test_markdown_matches_template_layout():
    md = build_model_links_markdown([
        {'name': 'my_lora.safetensors', 'url': 'https://civitai.com/api/download/models/9?token=s', 'directory': 'loras'},
        {'name': 'flux1-dev.safetensors', 'url': 'https://huggingface.co/a/b/resolve/main/flux1-dev.safetensors',
         'directory': 'diffusion_models'},
        {'name': 'styles\\private.safetensors', 'url': '', 'directory': 'loras'},
    ])
    expected = '\n'.join([
        '## Model links',
        '',
        '**diffusion_models**',
        '',
        '- [flux1-dev.safetensors](https://huggingface.co/a/b/resolve/main/flux1-dev.safetensors)',
        '',
        '**loras**',
        '',
        '- [my_lora.safetensors](https://civitai.com/api/download/models/9)',
        '- private.safetensors (no public link)',
        '',
        '## Model storage location',
        '',
        '```',
        '📂 ComfyUI/',
        '├── 📂 models/',
        '│   ├── 📂 diffusion_models/',
        '│   │   └── flux1-dev.safetensors',
        '│   └── 📂 loras/',
        '│       ├── my_lora.safetensors',
        '│       └── styles/private.safetensors',
        '```',
        '',
        '*Prepared with [Model Linker](https://github.com/kianxyzw/comfyui-model-linker)*',
    ])
    assert md == expected, md


def test_markdown_dedupes_and_escapes():
    md = build_model_links_markdown([
        {'name': 'a [v2].safetensors', 'url': 'https://h.co/a b.safetensors', 'directory': 'loras'},
        {'name': 'a [v2].safetensors', 'url': 'https://h.co/a b.safetensors', 'directory': 'loras'},
    ])
    assert md.count('- [a \\[v2\\].safetensors](https://h.co/a%20b.safetensors)') == 1


def test_lookup_links_prefers_existing_and_strips():
    calls = []

    def fake_chain(filename, category='', workflow_url='', sha256='', fetch_size=True):
        calls.append((filename, workflow_url))
        if workflow_url:
            return {'url': workflow_url, 'source': 'civitai', 'url_source': 'workflow', 'match_type': 'exact'}
        return None

    results = lookup_links([
        {'filename': 'a.safetensors', 'existing_url': 'https://civitai.com/api/download/models/1?token=x'},
        {'filename': 'b.safetensors'},
    ], fake_chain)
    assert results[0]['url'] == 'https://civitai.com/api/download/models/1'
    assert results[1] is None
    assert calls[0][1] == 'https://civitai.com/api/download/models/1'


def test_hash_path_guard():
    assert is_inside_model_dirs(INSIDE_FILE) is True
    assert is_inside_model_dirs(OUTSIDE_FILE) is False
    assert is_inside_model_dirs(os.path.join(MODELS_DIR, 'loras', '..', '..', os.path.basename(OUTSIDE_DIR),
                                             'secret.txt')) is False


def test_compute_sha256():
    assert compute_sha256(INSIDE_FILE) == '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'


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
