"""
Settings Module

Stores API tokens for HuggingFace and CivitAI on the server. Tokens never
go back to the browser: the UI only sees whether one is set and its last
four characters.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = 'model_linker_settings.json'

# provider -> (settings key, environment variable fallback)
PROVIDERS = {
    'huggingface': ('hf_token', 'HF_TOKEN'),
    'civitai': ('civitai_key', 'CIVITAI_API_TOKEN'),
}

_lock = threading.Lock()


def get_settings_path() -> str:
    """Settings live in ComfyUI's user directory, outside the extension folder."""
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, SETTINGS_FILENAME)


def load_settings() -> Dict[str, Any]:
    path = get_settings_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Model Linker: could not read settings: {e}")
        return {}


def save_settings(updates: Dict[str, Any]) -> None:
    """Merge updates into the settings file; empty strings remove a key."""
    with _lock:
        settings = load_settings()
        for key, value in updates.items():
            if value in (None, ''):
                settings.pop(key, None)
            else:
                settings[key] = value

        path = get_settings_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + '.tmp'
        # Owner-only permissions where the OS supports them (no-op on Windows)
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp_path, path)


def get_token(provider: str) -> Optional[str]:
    """Token for 'huggingface' or 'civitai': saved setting first, then env var."""
    if provider not in PROVIDERS:
        return None
    key, env_var = PROVIDERS[provider]
    token = load_settings().get(key) or os.environ.get(env_var)
    return token.strip() if isinstance(token, str) and token.strip() else None


def public_view() -> Dict[str, Any]:
    """What the UI may see: whether a token is set, where from, and a hint."""
    settings = load_settings()
    view = {}
    for provider, (key, env_var) in PROVIDERS.items():
        saved = settings.get(key)
        env = os.environ.get(env_var)
        token = saved or env
        view[provider] = {
            'set': bool(token),
            'source': 'settings' if saved else ('env' if env else None),
            'env_var': env_var,
            'hint': f'…{token[-4:]}' if token and len(token) > 8 else ('set' if token else ''),
        }
    return view


def provider_for_url(url: str) -> Optional[str]:
    if 'huggingface.co' in url or 'hf.co/' in url:
        return 'huggingface'
    if 'civitai.com' in url:
        return 'civitai'
    return None


def auth_headers_for_url(url: str) -> Dict[str, str]:
    """Authorization header for a download/search URL, if we hold a token for its host."""
    provider = provider_for_url(url or '')
    token = get_token(provider) if provider else None
    return {'Authorization': f'Bearer {token}'} if token else {}


def check_token(provider: str) -> Dict[str, Any]:
    """Validate the stored token against the provider's API."""
    token = get_token(provider)
    if not token:
        return {'ok': False, 'error': 'No token set'}

    if provider == 'huggingface':
        url = 'https://huggingface.co/api/whoami-v2'
    elif provider == 'civitai':
        url = 'https://civitai.com/api/v1/me'
    else:
        return {'ok': False, 'error': f'Unknown provider: {provider}'}

    try:
        response = requests.get(url, headers={'Authorization': f'Bearer {token}'}, timeout=10)
    except Exception as e:
        return {'ok': False, 'error': f'Could not reach {provider}: {e}'}

    if response.status_code in (401, 403):
        return {'ok': False, 'error': 'Token was rejected'}
    if response.status_code != 200:
        return {'ok': False, 'error': f'Unexpected response ({response.status_code})'}

    try:
        data = response.json()
    except ValueError:
        data = {}
    user = data.get('name') or data.get('username') or data.get('fullname')
    return {'ok': True, 'user': user}
