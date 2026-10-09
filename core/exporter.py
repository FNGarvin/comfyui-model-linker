"""
Exporter Module

Prepares a workflow for sharing: finds a public download link for every
model, builds a "Model links" note in the layout of ComfyUI's official
templates, and hashes local files to identify models the name lookup
can't find.
"""

import hashlib
import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

# Order of folders in the note, following ComfyUI's official templates;
# anything else follows alphabetically
CATEGORY_ORDER = [
    'checkpoints', 'diffusion_models', 'unet', 'text_encoders', 'clip', 'clip_vision',
    'vae', 'loras', 'controlnet', 'model_patches', 'upscale_models', 'embeddings',
    'style_models', 'ipadapter', 'audio_encoders', 'latent_upscale_models',
]

# Query parameters that carry credentials and must never end up in a
# shared workflow
SECRET_PARAMS = {'token', 'key', 'api_key', 'apikey', 'access_token', 'auth', 'hf_token', 'sig', 'signature'}

_hash_jobs: Dict[str, Dict[str, Any]] = {}
_hash_cache: Dict[tuple, str] = {}
_hash_lock = threading.Lock()


def strip_secrets(url: str) -> str:
    """Remove credential query parameters (token=, key=, ...) from a URL."""
    if not url or not isinstance(url, str):
        return ''
    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if parsed.scheme not in ('http', 'https'):
        return url
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if k.lower() not in SECRET_PARAMS]
    return urlunparse(parsed._replace(query=urlencode(query)))


def _category_sort_key(category: str):
    if category in CATEGORY_ORDER:
        return (0, CATEGORY_ORDER.index(category), '')
    return (1, 0, category)


def _display_path(name: str) -> List[str]:
    """Path parts of a model name, separators normalized."""
    return [part for part in name.replace('\\', '/').split('/') if part]


def _markdown_link_text(text: str) -> str:
    return text.replace('[', '\\[').replace(']', '\\]')


def _markdown_url(url: str) -> str:
    return url.replace(' ', '%20').replace('(', '%28').replace(')', '%29')


def build_model_links_markdown(models: List[Dict[str, Any]]) -> str:
    """
    Markdown for a "Model links" note, e.g.

        ## Model links

        **diffusion_models**

        - [flux1-dev.safetensors](https://huggingface.co/...)

        ## Model storage location

        ```
        📂 ComfyUI/
        ├── 📂 models/
        │   └── 📂 diffusion_models/
        │       └── flux1-dev.safetensors
        ```

    models: [{name, url, directory}], name may include a subfolder.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    for model in models:
        name = (model.get('name') or '').strip()
        directory = (model.get('directory') or 'unknown').strip() or 'unknown'
        if not name or (directory, name) in seen:
            continue
        seen.add((directory, name))
        groups.setdefault(directory, []).append(model)

    categories = sorted(groups, key=_category_sort_key)
    for category in categories:
        groups[category].sort(key=lambda m: m['name'].lower())

    lines = ['## Model links', '']
    for category in categories:
        lines += [f'**{category}**', '']
        for model in groups[category]:
            filename = _display_path(model['name'])[-1]
            url = strip_secrets(model.get('url') or '')
            if url:
                lines.append(f'- [{_markdown_link_text(filename)}]({_markdown_url(url)})')
            else:
                lines.append(f'- {filename} (no public link)')
        lines.append('')

    lines += ['## Model storage location', '', '```', '📂 ComfyUI/', '├── 📂 models/']
    for cat_index, category in enumerate(categories):
        last_category = cat_index == len(categories) - 1
        lines.append(f"│   {'└──' if last_category else '├──'} 📂 {category}/")
        child_prefix = '│   ' + ('    ' if last_category else '│   ')
        for model_index, model in enumerate(groups[category]):
            last_model = model_index == len(groups[category]) - 1
            lines.append(f"{child_prefix}{'└──' if last_model else '├──'} {'/'.join(_display_path(model['name']))}")
    lines += ['```', '', '*Prepared with [Model Linker](https://github.com/kianxyzw/comfyui-model-linker)*']
    return '\n'.join(lines)


def lookup_links(models: List[Dict[str, Any]], find_download_source) -> List[Optional[Dict[str, Any]]]:
    """
    Public download link for each model: [{url, source, match_type}] or None.

    A link already in the workflow wins; otherwise the regular download
    source chain is used (exact filename matches only).
    """
    results = []
    for model in models:
        filename = model.get('filename') or model.get('name') or ''
        existing = strip_secrets(model.get('existing_url') or '')
        try:
            source = find_download_source(
                filename,
                category=model.get('category', ''),
                workflow_url=existing,
                sha256=model.get('sha256', ''),
                fetch_size=False,
            )
        except Exception as e:
            logger.warning(f"Model Linker: link lookup failed for {filename}: {e}")
            source = None
        if source and source.get('url'):
            results.append({
                'url': strip_secrets(source['url']),
                'source': source.get('source', ''),
                'url_source': source.get('url_source', ''),
                'match_type': source.get('match_type', ''),
            })
        else:
            results.append(None)
    return results


def is_inside_model_dirs(path: str) -> bool:
    """True if path is a file under one of ComfyUI's model folders."""
    try:
        import folder_paths
    except ImportError:
        return False
    if not path or not os.path.isfile(path):
        return False
    real = os.path.normcase(os.path.realpath(path))
    for category, entry in folder_paths.folder_names_and_paths.items():
        if category in ('custom_nodes', 'configs'):
            continue
        for base in entry[0]:
            base_real = os.path.normcase(os.path.realpath(base))
            if real.startswith(base_real.rstrip(os.sep) + os.sep):
                return True
    return False


def start_hash_job(path: str, find_by_hash) -> str:
    """Hash a local model file in the background, then look the hash up."""
    job_id = uuid.uuid4().hex[:12]
    with _hash_lock:
        _hash_jobs[job_id] = {'status': 'hashing', 'progress': 0.0, 'sha256': None, 'result': None, 'error': None}

    def run():
        try:
            sha256 = compute_sha256(path, job_id)
            with _hash_lock:
                _hash_jobs[job_id].update({'status': 'looking_up', 'progress': 1.0, 'sha256': sha256})
            result = find_by_hash(sha256, os.path.basename(path))
            if result and result.get('url'):
                result = {
                    'url': strip_secrets(result['url']),
                    'source': result.get('source', ''),
                    'url_source': result.get('url_source', ''),
                    'match_type': 'hash',
                }
            else:
                result = None
            with _hash_lock:
                _hash_jobs[job_id].update({'status': 'done', 'result': result})
        except Exception as e:
            logger.warning(f"Model Linker: hashing {path} failed: {e}")
            with _hash_lock:
                _hash_jobs[job_id].update({'status': 'error', 'error': str(e)})

    threading.Thread(target=run, daemon=True).start()
    return job_id


def compute_sha256(path: str, job_id: Optional[str] = None) -> str:
    """SHA-256 of a file, cached by path, size and modification time."""
    stat = os.stat(path)
    cache_key = (os.path.normcase(os.path.realpath(path)), stat.st_size, stat.st_mtime)
    if cache_key in _hash_cache:
        return _hash_cache[cache_key]

    digest = hashlib.sha256()
    done = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if job_id and stat.st_size:
                with _hash_lock:
                    _hash_jobs[job_id]['progress'] = done / stat.st_size

    sha256 = digest.hexdigest()
    _hash_cache[cache_key] = sha256
    return sha256


def get_hash_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _hash_lock:
        job = _hash_jobs.get(job_id)
        return dict(job) if job else None
