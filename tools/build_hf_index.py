"""
Build metadata/hf-index.json: every model file in the HuggingFace repos of
authors whose files ComfyUI workflows reference most (Comfy-Org's split
files, Kijai's conversions), with size and SHA-256.

Model Linker looks missing files up here before calling the HuggingFace
API, which finds files in subfolders (split_files/...) that search misses
and identifies files by hash.

Run from the repo root:  python tools/build_hf_index.py
Optional: HF_TOKEN env var for higher API rate limits.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests

AUTHORS = ['Comfy-Org', 'Kijai']
MODEL_EXTENSIONS = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.onnx', '.sft', '.pt2')
API = 'https://huggingface.co/api'
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'metadata', 'hf-index.json')
INDEX_VERSION = 1

session = requests.Session()
if os.environ.get('HF_TOKEN'):
    session.headers['Authorization'] = f"Bearer {os.environ['HF_TOKEN']}"


def get_json(url: str, attempts: int = 4):
    for attempt in range(attempts):
        response = session.get(url, timeout=60)
        if response.status_code == 429 and attempt < attempts - 1:
            time.sleep(min(float(response.headers.get('Retry-After', 5)), 60))
            continue
        response.raise_for_status()
        return response.json()


def list_repos(author: str) -> list:
    repos = get_json(f'{API}/models?author={quote(author)}&limit=1000')
    return sorted(r['id'] for r in repos if not r.get('private') and not r.get('gated'))


def list_model_files(repo_id: str) -> list:
    try:
        tree = get_json(f'{API}/models/{repo_id}/tree/main?recursive=true')
    except requests.HTTPError as e:
        print(f'  skip {repo_id}: {e}', file=sys.stderr)
        return []

    files = []
    for entry in tree:
        path = entry.get('path', '')
        if entry.get('type') != 'file' or not path.lower().endswith(MODEL_EXTENSIONS):
            continue
        lfs = entry.get('lfs') or {}
        files.append({
            'repo_id': repo_id,
            'path': path,
            'filename': path.rsplit('/', 1)[-1],
            'size': lfs.get('size', entry.get('size')),
            'sha256': (lfs.get('oid') or '').lower(),
        })
    return files


def main():
    repos = []
    for author in AUTHORS:
        author_repos = list_repos(author)
        print(f'{author}: {len(author_repos)} repos')
        repos.extend(author_repos)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(list_model_files, repos))

    files = sorted((f for repo_files in results for f in repo_files),
                   key=lambda f: (f['repo_id'], f['path']))
    index = {
        'version': INDEX_VERSION,
        'authors': AUTHORS,
        'files': files,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(index, f, indent=0, ensure_ascii=False)
        f.write('\n')
    print(f'Wrote {len(files)} files from {len(repos)} repos to {OUTPUT}')


if __name__ == '__main__':
    main()
