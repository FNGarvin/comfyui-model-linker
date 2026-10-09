"""
Download Source Chain

Finds where a model file can be downloaded, trying sources from most to
least certain. Shared by workflow analysis (missing models) and export
(links for every model). All calls are blocking - run them off the event
loop (asyncio.to_thread).
"""

import logging
from typing import Any, Dict, Optional

import requests

from .settings import get_token
from .sources.popular import get_popular_model_url
from .sources.model_list import search_model_list
from .sources.huggingface import search_huggingface_for_file
from .sources.civitai import search_civitai_for_file, search_civitai_by_hash
from .sources import hf_index, civarchive

logger = logging.getLogger(__name__)


def source_from_url(url: str) -> str:
    if 'huggingface.co' in url:
        return 'huggingface'
    if 'civitai.com' in url:
        return 'civitai'
    return 'workflow'


def find_by_hash(sha256: str, filename: str = '') -> Optional[Dict[str, Any]]:
    """Identify a file by SHA-256: bundled HF index, CivitAI, then CivArchive."""
    if not sha256:
        return None

    result = hf_index.lookup_by_sha256(sha256)
    if result:
        return result

    civitai_result = search_civitai_by_hash(sha256, api_key=get_token('civitai'))
    if civitai_result and civitai_result.get('download_url'):
        return {
            'source': 'civitai',
            'url_source': 'civitai_hash',
            'url': civitai_result['download_url'],
            'page_url': civitai_result.get('url', ''),
            'filename': civitai_result.get('filename') or filename,
            'name': civitai_result.get('name', ''),
            'type': civitai_result.get('type', ''),
            'size': civitai_result.get('size'),
            'match_type': 'hash',
        }

    return civarchive.lookup_by_sha256(sha256, filename)


def find_download_source(
    filename: str,
    category: str = '',
    workflow_url: str = '',
    workflow_directory: str = '',
    sha256: str = '',
    fetch_size: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Best download source for a model, or None.

    Order: URL from the workflow itself, curated list, ComfyUI-Manager model
    list, bundled HuggingFace index, hash lookups (when the workflow carries
    a SHA-256), then HuggingFace and CivitAI search. Only exact filename or
    hash matches are returned - a download must be the right file.
    """
    filename = (filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    if not filename:
        return None
    default_directory = workflow_directory or category or 'checkpoints'

    # 0. URL from the workflow (highest priority - the author's own link)
    if workflow_url:
        file_size = None
        if fetch_size:
            try:
                head = requests.head(workflow_url, allow_redirects=True, timeout=5)
                if head.status_code == 200:
                    file_size = int(head.headers.get('content-length', 0)) or None
            except Exception:
                pass  # Size unknown is fine
        return {
            'source': source_from_url(workflow_url),
            'url': workflow_url,
            'filename': filename,
            'directory': default_directory,
            'match_type': 'exact',
            'url_source': 'workflow',
            'size': file_size
        }

    # 1. Curated popular models (always exact)
    popular_info = get_popular_model_url(filename)
    if popular_info:
        return {
            'source': 'popular',
            'url': popular_info.get('url'),
            'filename': filename,
            'type': popular_info.get('type'),
            'directory': popular_info.get('directory') or default_directory,
            'match_type': 'exact'
        }

    # 2. ComfyUI-Manager model database (exact only for downloads)
    model_list_result = search_model_list(filename, exact_only=True)
    if model_list_result:
        return {
            'source': 'model_list',
            'url': model_list_result.get('url'),
            'filename': model_list_result.get('filename'),
            'name': model_list_result.get('name'),
            'type': model_list_result.get('type'),
            'directory': model_list_result.get('directory') or default_directory,
            'size': model_list_result.get('size'),
            'match_type': model_list_result.get('match_type'),
            'confidence': model_list_result.get('confidence')
        }

    # 3. Bundled index of Comfy-Org / Kijai HuggingFace files
    index_result = hf_index.lookup_by_filename(filename)
    if index_result:
        return {**index_result, 'name': index_result['repo_id'], 'directory': default_directory}

    # 4. Identify by hash when the workflow carries one
    hash_result = find_by_hash(sha256, filename)
    if hash_result:
        return {**hash_result, 'directory': default_directory}

    # 5. HuggingFace search (exact filename)
    hf_result = search_huggingface_for_file(filename, token=get_token('huggingface'), exact_only=True)
    if hf_result:
        return {
            'source': 'huggingface',
            'url': hf_result.get('url'),
            'filename': hf_result.get('filename'),
            'name': hf_result.get('repo_id', ''),
            'size': hf_result.get('size', ''),
            'directory': default_directory,
            'match_type': hf_result.get('match_type', 'exact')
        }

    # 6. CivitAI search (exact filename)
    civitai_result = search_civitai_for_file(filename, api_key=get_token('civitai'), exact_only=True)
    if civitai_result:
        return {
            'source': 'civitai',
            'url': civitai_result.get('download_url'),  # CivitAI uses download_url
            'filename': civitai_result.get('filename'),
            'name': civitai_result.get('name', ''),
            'size': civitai_result.get('size', ''),
            'directory': default_directory,
            'match_type': civitai_result.get('match_type', 'exact')
        }

    return None
