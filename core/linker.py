"""
Core Linker Module

Integrates all components to provide high-level API for model linking.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import unquote

from .scanner import get_model_files
from .workflow_analyzer import analyze_workflow_models, identify_missing_models, group_models_by_file
from .matcher import find_matches, extract_filename
from .workflow_updater import update_workflow_nodes

logger = logging.getLogger(__name__)

# Regex patterns for URL extraction (matches HuggingFace and CivitAI URLs)
URL_PATTERN = re.compile(r'(https?://(?:huggingface\.co|civitai\.com)[^\s"\'<>\)\\]+)')

# Model file extensions to look for
MODEL_EXTENSIONS = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.onnx', '.gguf')


def extract_workflow_urls(workflow_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract model URLs from workflow JSON.
    
    Sources:
    1. node.properties.models array - contains {name, url, directory} and
       optionally {hash, hash_type} (ComfyUI's model metadata format)
    2. Regex extraction from workflow JSON string - finds HuggingFace/CivitAI URLs
    3. extra.model_resolver_hashes - SHA-256s embedded by ComfyUI-Model-Resolver
    
    Args:
        workflow_json: Complete workflow JSON dictionary
        
    Returns:
        Dict mapping model filename -> {url, directory, source, sha256?}
    """
    url_map = {}
    
    # Convert to string for regex search
    workflow_str = json.dumps(workflow_json)
    
    # Collect all nodes including from subgraphs
    all_nodes = list(workflow_json.get('nodes', []))
    definitions = workflow_json.get('definitions', {})
    subgraphs = definitions.get('subgraphs', [])
    for subgraph in subgraphs:
        subgraph_nodes = subgraph.get('nodes', [])
        all_nodes.extend(subgraph_nodes)
    
    # 1. Extract from node.properties.models (authoritative source)
    for node in all_nodes:
        node_type = node.get('type', '')
        properties = node.get('properties', {})
        models_list = properties.get('models', [])
        
        for model_info in models_list:
            if isinstance(model_info, dict):
                name = model_info.get('name', '')
                url = model_info.get('url', '')
                directory = model_info.get('directory', '')
                sha256 = ''
                if str(model_info.get('hash_type') or 'sha256').lower() == 'sha256':
                    sha256 = _clean_sha256(model_info.get('hash'))

                if name and name not in url_map:
                    url_map[name] = {
                        'url': url,
                        'directory': directory,
                        'node_type': node_type,
                        'source': 'node_properties'
                    }
                    if sha256:
                        url_map[name]['sha256'] = sha256
    
    # 1b. SHA-256s embedded by ComfyUI-Model-Resolver ("Embed workflow hashes")
    resolver_hashes = (workflow_json.get('extra') or {}).get('model_resolver_hashes') or {}
    for entry in resolver_hashes.get('models') or []:
        if not isinstance(entry, dict):
            continue
        sha256 = _clean_sha256(entry.get('sha256'))
        path = entry.get('path')
        if not sha256 or not isinstance(path, str) or not path:
            continue
        name = path.replace('\\', '/').rsplit('/', 1)[-1]
        url_map.setdefault(name, {'url': '', 'directory': '', 'source': 'model_resolver_hashes'})
        url_map[name].setdefault('sha256', sha256)

    # 2. Extract URLs via regex from workflow JSON
    urls_found = URL_PATTERN.findall(workflow_str)
    
    # Clean URLs (remove trailing characters that may have been captured)
    cleaned_urls = []
    for url in urls_found:
        url = url.split(')')[0].replace('\\n', '').replace('\n', '').strip()
        if url:
            cleaned_urls.append(url)
    
    # 3. Extract model filenames via regex
    model_pattern = re.compile(r'([\w\-\.%]+\.(?:safetensors|ckpt|pt|pth|bin|onnx|gguf))', re.IGNORECASE)
    model_files_raw = model_pattern.findall(workflow_str)
    
    # Clean and decode filenames
    model_files = set()
    model_name_map = {}  # decoded -> original
    
    for model in model_files_raw:
        cleaned = model.strip()
        if cleaned and cleaned[0].isalnum():
            try:
                decoded = unquote(cleaned)
            except Exception:
                decoded = cleaned
            model_files.add(decoded)
            model_name_map[decoded] = cleaned
    
    # 4. Match URLs to model filenames
    for model in model_files:
        # Skip if already found in node.properties.models
        if model in url_map and url_map[model].get('url'):
            continue
        
        original_name = model_name_map.get(model, model)
        
        for url in cleaned_urls:
            # Check decoded name in URL
            if model in url:
                if model not in url_map:
                    url_map[model] = {'url': url, 'directory': '', 'source': 'regex'}
                elif not url_map[model].get('url'):
                    url_map[model]['url'] = url
                    url_map[model]['source'] = 'regex'
                break
            # Check original (possibly URL-encoded) name in URL
            if original_name in url:
                if model not in url_map:
                    url_map[model] = {'url': url, 'directory': '', 'source': 'regex'}
                elif not url_map[model].get('url'):
                    url_map[model]['url'] = url
                    url_map[model]['source'] = 'regex'
                break
            # Check without extension
            model_base = os.path.splitext(model)[0]
            if model_base in url or unquote(model_base) in url:
                if model not in url_map:
                    url_map[model] = {'url': url, 'directory': '', 'source': 'regex'}
                elif not url_map[model].get('url'):
                    url_map[model]['url'] = url
                    url_map[model]['source'] = 'regex'
                break
    
    return url_map


def _clean_sha256(value: Any) -> str:
    """Lowercase SHA-256 hex digest, or '' if value isn't one."""
    if isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value.strip()):
        return value.strip().lower()
    return ''


def parse_huggingface_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract HuggingFace repo and path from URL.
    
    Args:
        url: HuggingFace URL
        
    Returns:
        Tuple of (repo_id, file_path) or (None, None) if not valid
    """
    if not url or 'huggingface.co' not in url:
        return None, None
    
    # Pattern: https://huggingface.co/user/repo/resolve/main/path/to/file.safetensors
    match = re.match(r'https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+)', url)
    if match:
        return match.group(1), match.group(2)
    
    return None, None


def attach_workflow_urls(models: List[Dict[str, Any]], workflow_urls: Dict[str, Dict[str, Any]]) -> None:
    """Copy URL / directory / SHA-256 found in the workflow onto each model entry."""
    for model in models:
        original_path = model.get('original_path', '')
        url_info = workflow_urls.get(original_path) or workflow_urls.get(extract_filename(original_path))
        if not url_info:
            continue
        model['workflow_url'] = url_info.get('url', '')
        model['workflow_directory'] = url_info.get('directory', '')
        model['url_source'] = url_info.get('source', '')
        if url_info.get('sha256'):
            model['workflow_sha256'] = url_info['sha256']


def analyze_and_find_matches(
    workflow_json: Dict[str, Any],
    similarity_threshold: float = 0.0,
    max_matches_per_model: int = 10,
    include_resolved: bool = False
) -> Dict[str, Any]:
    """
    Main entry point: analyze workflow and find matches for missing models.
    
    Args:
        workflow_json: Complete workflow JSON dictionary
        similarity_threshold: Minimum similarity score (0.0 to 1.0) for matches
        max_matches_per_model: Maximum number of matches to return per missing model
        
    Returns:
        Dictionary with analysis results:
        {
            'missing_models': [
                {
                    'node_id': node ID,
                    'node_type': node type,
                    'widget_index': widget index,
                    'original_path': original path from workflow,
                    'category': model category,
                    'workflow_url': URL from workflow if found,
                    'workflow_directory': directory from workflow if found,
                    'matches': [
                        {
                            'model': model dict from scanner,
                            'filename': model filename,
                            'similarity': similarity score (0.0-1.0),
                            'confidence': confidence percentage (0-100)
                        },
                        ...
                    ]
                },
                ...
            ],
            'total_missing': count of missing models,
            'total_models_analyzed': count of all models in workflow
        }
    """
    # Extract URLs from workflow (node.properties.models + regex)
    workflow_urls = extract_workflow_urls(workflow_json)
    logger.debug(f"Extracted {len(workflow_urls)} URLs from workflow")
    
    # Analyze workflow to find all model references
    all_model_refs = analyze_workflow_models(workflow_json)
    
    # Get available models
    available_models = get_model_files()
    
    # Identify missing models
    missing_models = identify_missing_models(all_model_refs, available_models)
    
    # Enrich missing models with workflow URLs
    attach_workflow_urls(missing_models, workflow_urls)
    
    # Find matches for each missing model
    missing_with_matches = []
    for missing in missing_models:
        original_path = missing.get('original_path', '')
        
        # Filter available models by category if known
        # IMPORTANT: If category is 'unknown', we still try to find the right category
        # by using node type hints
        category = missing.get('category')
        
        # If category is unknown, try to use node type to infer category
        if not category or category == 'unknown':
            from .workflow_analyzer import NODE_TYPE_TO_CATEGORY_HINTS
            node_type = missing.get('node_type', '')
            category = NODE_TYPE_TO_CATEGORY_HINTS.get(node_type, 'unknown')
        
        candidates = available_models
        if category and category != 'unknown':
            # Prioritize models from the same category
            candidates = [m for m in available_models if m.get('category') == category]
            # Also include other categories as fallback
            candidates.extend([m for m in available_models if m.get('category') != category])

        # Categories the node can actually load from (issue #3): prefer the
        # analyzer's introspected list, fall back to the single category hint
        expected_categories = missing.get('expected_categories')
        if not expected_categories and category and category != 'unknown':
            expected_categories = [category]

        # Find matches
        matches = find_matches(
            original_path,
            candidates,
            threshold=similarity_threshold,
            max_results=max_matches_per_model,
            expected_categories=expected_categories
        )
        
        # Deduplicate matches by absolute path - same physical file should only appear once
        # This handles cases where the same file exists in multiple base directories
        # or has different relative_paths but is the same file
        seen_absolute_paths = {}
        deduplicated_matches = []
        for match in matches:
            model_dict = match['model']
            absolute_path = model_dict.get('path', '')
            
            # Normalize absolute path for comparison
            if absolute_path:
                absolute_path = os.path.normpath(absolute_path)
            
            # If we haven't seen this absolute path, add it
            if absolute_path not in seen_absolute_paths:
                seen_absolute_paths[absolute_path] = match
                deduplicated_matches.append(match)
            else:
                # If we've seen this absolute path before, replace with better match if confidence is higher
                existing_match = seen_absolute_paths[absolute_path]
                if match['confidence'] > existing_match['confidence']:
                    # Replace with better match
                    idx = deduplicated_matches.index(existing_match)
                    deduplicated_matches[idx] = match
                    seen_absolute_paths[absolute_path] = match
        
        missing_with_matches.append({
            **missing,
            'matches': deduplicated_matches
        })
    
    result = {
        'missing_models': missing_with_matches,
        'total_missing': len(missing_with_matches),
        'total_models_analyzed': len(all_model_refs)
    }

    if include_resolved:
        # Grouped resolved references for the "All models" browse/swap view
        result['resolved_models'] = group_models_by_file(all_model_refs, exists_filter=True)
        # Existing links let the Export tab reuse them
        attach_workflow_urls(result['resolved_models'], workflow_urls)

    return result


def apply_resolution(
    workflow_json: Dict[str, Any],
    resolutions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply model resolutions to workflow.
    
    Args:
        workflow_json: Workflow JSON dictionary (will be modified)
        resolutions: List of resolution dictionaries:
            {
                'node_id': node ID,
                'widget_index': widget index,
                'resolved_path': absolute path to resolved model,
                'category': model category (optional),
                'resolved_model': model dict from scanner (optional)
            }
            
    Returns:
        Updated workflow JSON dictionary
    """
    # Prepare mappings for workflow_updater
    mappings = []
    for resolution in resolutions:
        mapping = {
            'node_id': resolution.get('node_id'),
            'widget_index': resolution.get('widget_index'),
            'resolved_path': resolution.get('resolved_path'),
            'category': resolution.get('category'),
            'resolved_model': resolution.get('resolved_model'),
            'subgraph_id': resolution.get('subgraph_id'),  # Include subgraph_id for subgraph nodes
            'is_top_level': resolution.get('is_top_level'),  # True for top-level nodes, False for nodes in subgraph definitions
            # Filename nested in a dict/list widget (rgthree, LoraManager)
            'nested_key': resolution.get('nested_key'),
            'list_index': resolution.get('list_index'),
            'text_widget_index': resolution.get('text_widget_index')
        }
        
        # If resolved_model provided, extract path if needed
        if 'resolved_model' in resolution and resolution['resolved_model']:
            resolved_model = resolution['resolved_model']
            if 'path' in resolved_model and not mapping.get('resolved_path'):
                mapping['resolved_path'] = resolved_model['path']
            if 'base_directory' in resolved_model:
                mapping['base_directory'] = resolved_model['base_directory']
        
        mappings.append(mapping)
    
    # Update workflow
    updated_workflow = update_workflow_nodes(workflow_json, mappings)
    
    return updated_workflow


def get_resolution_summary(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary of missing models and matches without applying resolutions.
    
    This is a convenience method that calls analyze_and_find_matches with defaults.
    
    Args:
        workflow_json: Complete workflow JSON dictionary
        
    Returns:
        Same format as analyze_and_find_matches
    """
    return analyze_and_find_matches(workflow_json)

