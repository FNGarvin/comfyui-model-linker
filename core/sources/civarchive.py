"""
CivArchive source.

civarchive.com indexes CivitAI models by file SHA-256, including ones
deleted from CivitAI, along with mirrors on HuggingFace. Only used when
the workflow carries a model's hash.
"""

import logging
import re
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

API_URL = 'https://civarchive.com/api/sha256/{sha256}'
SHA256_PATTERN = re.compile(r'^[0-9a-fA-F]{64}$')

_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def _get(url: str) -> Optional[requests.Response]:
    """GET with one retry on rate limiting (Retry-After capped at 3 s)."""
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=10, headers={'Accept': 'application/json'})
        except requests.RequestException as e:
            logger.debug(f"CivArchive request failed: {e}")
            return None
        if response.status_code == 429 and attempt == 0:
            try:
                delay = float(response.headers.get('Retry-After', 1))
            except ValueError:
                delay = 1.0
            time.sleep(min(max(delay, 0.5), 3.0))
            continue
        return response
    return None


def lookup_by_sha256(sha256: str, filename: str = '') -> Optional[Dict[str, Any]]:
    """
    Find a downloadable copy of the file with this SHA-256.

    Live, ungated, free copies only. A HuggingFace mirror is preferred over
    CivitAI's own link, which may need an API key or have been deleted.
    """
    if not sha256 or not SHA256_PATTERN.match(sha256) or set(sha256) == {'0'}:
        return None
    sha256 = sha256.lower()
    if sha256 in _cache:
        return _cache[sha256]

    result = None
    response = _get(API_URL.format(sha256=sha256))
    if response is not None and response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            data = {}
        files = [f for f in data.get('files') or []
                 if isinstance(f, dict) and f.get('url')
                 and not f.get('deletedAt') and not f.get('is_gated') and not f.get('is_paid')]
        # Same filename first, then HuggingFace mirrors before CivitAI
        files.sort(key=lambda f: (f.get('filename') != filename, f.get('source') == 'civitai'))
        if files:
            best = files[0]
            model = data.get('model') if isinstance(data.get('model'), dict) else {}
            result = {
                'source': best.get('source') or 'civarchive',
                'url_source': 'civarchive',
                'url': best['url'],
                'filename': best.get('filename') or filename,
                'name': model.get('name', ''),
                'type': model.get('type', ''),
                'sha256': sha256,
                'match_type': 'hash',
            }

    _cache[sha256] = result
    return result
