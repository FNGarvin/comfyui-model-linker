"""
HuggingFace Source Module

Search and download models from HuggingFace Hub.
"""

import os
import re
import logging
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, quote, unquote

logger = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api"

# Cache for search results
_search_cache: Dict[str, Any] = {}


def parse_huggingface_url(url: str) -> Optional[Dict[str, str]]:
    """
    Parse a HuggingFace URL to extract repo and filename.
    
    Handles formats:
    - https://huggingface.co/user/repo/resolve/main/file.safetensors
    - https://huggingface.co/user/repo/blob/main/file.safetensors
    - hf://user/repo/file.safetensors
    
    Returns:
        Dictionary with 'repo' and 'filename' keys, or None if not HF URL
    """
    if url.startswith('hf://'):
        parts = url[5:].split('/', 2)
        if len(parts) >= 3:
            return {
                'repo': unquote(f"{parts[0]}/{parts[1]}"),
                'filename': unquote(parts[2])
            }
        return None
    
    parsed = urlparse(url)
    if 'huggingface.co' not in parsed.netloc:
        return None
    
    match = re.match(r'^/([^/]+/[^/]+)/(resolve|blob)/([^/]+)/(.+)$', parsed.path)
    if match:
        return {
            'repo': unquote(match.group(1)),
            'branch': unquote(match.group(3)),
            'filename': unquote(match.group(4))
        }
    
    return None


def get_huggingface_download_url(repo: str, filename: str, branch: str = "main") -> str:
    """Generate a direct download URL for a HuggingFace file."""
    return f"https://huggingface.co/{repo}/resolve/{branch}/{quote(filename)}"


def clean_filename_for_search(filename: str) -> str:
    """
    Clean up filename for better search results.
    Remove common suffixes that might prevent matches.
    """
    base = os.path.splitext(filename)[0]
    # Remove common precision/format suffixes
    base = re.sub(r'[-_]?(fp16|fp32|fp8|bf16|e4m3fn|scaled|pruned|emaonly|q4|q8).*$', '', base, flags=re.IGNORECASE)
    return base


def build_search_queries(filename: str) -> List[str]:
    """
    Search queries for a filename: its stem, then the stem cut at the first
    precision/quant token (flux1-dev-fp8_e4m3fn -> flux1-dev). Repo names
    rarely carry those suffixes, so the shorter query finds the repo.
    """
    from ..matcher import is_variant_token

    stem = os.path.splitext(filename.replace('\\', '/').rsplit('/', 1)[-1])[0]
    queries = [stem]
    for token in re.finditer(r'[^-_.\s]+', stem):
        if is_variant_token(token.group().lower()):
            cut = stem[:token.start()].rstrip('-_. ')
            if cut:
                queries.append(cut)
            break
    seen = set()
    return [q for q in queries if q and not (q.lower() in seen or seen.add(q.lower()))]


def search_huggingface_for_file(
    filename: str,
    token: Optional[str] = None,
    exact_only: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Search HuggingFace for a specific model file.
    Returns the first repo that contains a matching file.

    Args:
        filename: Filename to search for
        token: Optional HF token
        exact_only: If True, only return exact filename matches (for downloads).
                   If False, also try partial matching (for local file resolution).

    Returns:
        Dict with url, repo, filename if found, None otherwise
    """
    global _search_cache

    cache_key = f"hf_{filename}_exact{exact_only}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    target_name = filename.replace('\\', '/').rsplit('/', 1)[-1]
    filename_base = os.path.splitext(target_name)[0].lower()

    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    checked_repos = set()
    partial_result = None
    try:
        for query in build_search_queries(target_name):
            # Search for repos matching this query
            search_url = f"{HF_API_URL}/models?search={quote(query)}&limit=10"
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.debug(f"HuggingFace search returned {response.status_code}")
                continue

            for repo in response.json():
                repo_id = repo.get('id', '')
                if not repo_id or repo_id in checked_repos:
                    continue
                checked_repos.add(repo_id)

                # List every file, including subfolders (split_files/...)
                files_url = f"{HF_API_URL}/models/{repo_id}/tree/main?recursive=true"
                try:
                    files_response = requests.get(files_url, headers=headers, timeout=10)
                    if files_response.status_code != 200:
                        continue
                    files = files_response.json()
                except Exception as e:
                    logger.debug(f"Error checking repo {repo_id}: {e}")
                    continue

                for file_info in files:
                    if file_info.get('type', 'file') != 'file':
                        continue
                    file_path = file_info.get('path', '')
                    file_name = os.path.basename(file_path)
                    size = (file_info.get('lfs') or {}).get('size', file_info.get('size'))

                    if file_name == target_name:
                        result = {
                            'source': 'huggingface',
                            'repo_id': repo_id,
                            'filename': file_name,
                            'path': file_path,
                            'url': get_huggingface_download_url(repo_id, file_path),
                            'size': size,
                            'match_type': 'exact'
                        }
                        _search_cache[cache_key] = result
                        logger.info(f"Found {filename} on HuggingFace: {repo_id}")
                        return result

                    # Partial matches only count when no exact match turns
                    # up anywhere - and never for downloads (exact_only)
                    if not exact_only and partial_result is None:
                        file_base = os.path.splitext(file_name)[0].lower()
                        if (filename_base in file_base or file_base in filename_base) and \
                                file_path.endswith(('.safetensors', '.ckpt')):
                            partial_result = {
                                'source': 'huggingface',
                                'repo_id': repo_id,
                                'filename': file_name,
                                'path': file_path,
                                'url': get_huggingface_download_url(repo_id, file_path),
                                'size': size,
                                'match_type': 'partial'
                            }

        if partial_result:
            logger.info(f"Found similar file for {filename} on HuggingFace: "
                        f"{partial_result['repo_id']}/{partial_result['path']}")
        _search_cache[cache_key] = partial_result
        return partial_result

    except Exception as e:
        logger.error(f"HuggingFace search error: {e}")
        return None


def search_huggingface(
    query: str,
    model_type: Optional[str] = None,
    limit: int = 10,
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub for models (general search).
    Returns repos that might be relevant, not guaranteed to have exact file.
    """
    results = []
    
    try:
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        params = {
            'search': query,
            'limit': limit,
            'full': 'true'
        }
        
        response = requests.get(
            f"{HF_API_URL}/models",
            params=params,
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            models = response.json()
            
            for model in models:
                repo_id = model.get('id', '')
                
                results.append({
                    'source': 'huggingface',
                    'repo': repo_id,
                    'name': model.get('modelId', repo_id),
                    'downloads': model.get('downloads', 0),
                    'likes': model.get('likes', 0),
                    'url': f"https://huggingface.co/{repo_id}"
                })
                
    except Exception as e:
        logger.error(f"HuggingFace search error: {e}")
    
    return results


def get_repo_files(
    repo: str,
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get list of model files in a HuggingFace repo."""
    files = []
    
    try:
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        response = requests.get(
            f"{HF_API_URL}/models/{repo}/tree/main",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            items = response.json()
            
            for item in items:
                path = item.get('path', '')
                if path.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth', '.onnx')):
                    files.append({
                        'filename': os.path.basename(path),
                        'path': path,
                        'url': get_huggingface_download_url(repo, path),
                        'size': item.get('size')
                    })
                    
    except Exception as e:
        logger.error(f"Error getting repo files: {e}")
    
    return files
