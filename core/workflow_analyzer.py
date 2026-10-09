"""
Workflow Analyzer Module

Extracts model references from workflow JSON and identifies missing models.
"""

import os
import re
import sys
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple

# Import folder_paths lazily - it may not be available until ComfyUI is initialized
try:
    import folder_paths
except ImportError:
    folder_paths = None
    logging.warning("Model Linker: folder_paths not available yet - will retry later")


# Common model file extensions
MODEL_EXTENSIONS = {'.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft', '.onnx'}

# Mapping of common node types to their expected model category
# This is used as hints but we don't rely solely on this
# UNETLoader uses 'diffusion_models' category (folder_paths maps 'unet' to 'diffusion_models')
NODE_TYPE_TO_CATEGORY_HINTS = {
    'CheckpointLoaderSimple': 'checkpoints',
    'CheckpointLoader': 'checkpoints',
    'unCLIPCheckpointLoader': 'checkpoints',
    'VAELoader': 'vae',
    'LoraLoader': 'loras',
    'LoraLoaderModelOnly': 'loras',
    'UNETLoader': 'diffusion_models',  # UNETLoader uses diffusion_models category
    'ControlNetLoader': 'controlnet',
    'ControlNetLoaderAdvanced': 'controlnet',
    'CLIPVisionLoader': 'clip_vision',
    'UpscaleModelLoader': 'upscale_models',
    'HypernetworkLoader': 'hypernetworks',
    'EmbeddingLoader': 'embeddings',
    'StyleModelLoader': 'style_models',
    'DiffusersLoader': 'diffusers',
}


def is_model_filename(value: Any) -> bool:
    """
    Check if a value looks like a model filename.

    Args:
        value: The value to check

    Returns:
        True if it looks like a model filename
    """
    if not isinstance(value, str):
        return False

    # Check if it ends with a model extension
    _, ext = os.path.splitext(value.lower())
    return ext in MODEL_EXTENSIONS


# Cache of node_type -> derived model field info (or None when nothing
# could be derived). Populated lazily; a failed introspection is cached too
# so misbehaving custom nodes are only probed once per session.
#   {'all': [categories], 'by_widget_index': {idx: [categories]},
#    'widget_count': number of widgets the node's inputs produce}
_NODE_MODEL_CATEGORIES_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

# Marker filenames handed out while tracing INPUT_TYPES(); they end in a
# model extension so filters like `if f.endswith('.safetensors')` keep them
_SENTINEL_PREFIX = '__model_linker_category__'
_SENTINEL_PATTERN = re.compile(re.escape(_SENTINEL_PREFIX) + r'(\d+)__\.safetensors')
_WIDGET_INPUT_TYPES = {'INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'}
# The frontend adds a control_after_generate widget after these inputs
_SEED_INPUT_NAMES = {'seed', 'noise_seed'}
_trace_lock = threading.RLock()


def get_node_model_field_info(node_type: str) -> Optional[Dict[str, Any]]:
    """
    Work out which inputs of a node type are model fields and which folder
    category each one loads from (issue #5).

    Custom loader nodes (Nunchaku, Hy3D, IPAdapter, GGUF, ...) build their
    file combos by calling folder_paths.get_filename_list() inside
    INPUT_TYPES(). We call INPUT_TYPES() with that function swapped for a
    tracer returning one marker filename per call; wherever a marker shows
    up in a combo, that combo loads from the traced category. This works
    even when the folder is empty or the node merges/filters lists. Combos
    filled some other way fall back to matching options against folders.
    """
    if node_type in _NODE_MODEL_CATEGORIES_CACHE:
        return _NODE_MODEL_CATEGORIES_CACHE[node_type]

    info = None
    try:
        import nodes as comfy_nodes
        node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_type)
        if node_class is not None:
            info = _derive_model_fields(node_class)
    except Exception as e:
        logging.debug(f"Model Linker: could not introspect {node_type}: {e}")

    _NODE_MODEL_CATEGORIES_CACHE[node_type] = info
    return info


def get_node_model_categories(node_type: str) -> Optional[List[str]]:
    """
    All model folder categories a node type loads from.

    Returns:
        List of category names, or None if nothing could be derived.
    """
    info = get_node_model_field_info(node_type)
    return info['all'] if info else None


def _trace_input_types(node_class) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Call node_class.INPUT_TYPES() with folder_paths.get_filename_list traced.

    Returns (input_types, {marker filename: category}). Only calls from this
    thread are traced; other threads (e.g. a prompt being validated) still
    get real file lists while the patch is in place.
    """
    global folder_paths
    if folder_paths is None:
        try:
            import folder_paths as fp
            folder_paths = fp
        except ImportError:
            return node_class.INPUT_TYPES(), {}

    original = folder_paths.get_filename_list
    tracer_thread = threading.get_ident()
    markers: Dict[str, str] = {}

    def traced_get_filename_list(folder_name, *args, **kwargs):
        if threading.get_ident() != tracer_thread:
            return original(folder_name, *args, **kwargs)
        category = folder_name
        map_legacy = getattr(folder_paths, 'map_legacy', None)
        if callable(map_legacy):
            category = map_legacy(folder_name)
        marker = f"{_SENTINEL_PREFIX}{len(markers)}__.safetensors"
        markers[marker] = category
        return [marker]

    # Nodes that did `from folder_paths import get_filename_list` hold their
    # own reference, so also patch it in the globals of the defining code
    namespaces = []
    for attr in ('INPUT_TYPES', 'define_schema'):
        func = getattr(getattr(node_class, attr, None), '__func__', None)
        func_globals = getattr(func, '__globals__', None)
        if isinstance(func_globals, dict):
            namespaces.append(func_globals)
    module = sys.modules.get(getattr(node_class, '__module__', ''))
    if module is not None:
        namespaces.append(module.__dict__)

    with _trace_lock:
        patched = []
        folder_paths.get_filename_list = traced_get_filename_list
        try:
            for namespace in namespaces:
                for name, value in list(namespace.items()):
                    if value is original:
                        namespace[name] = traced_get_filename_list
                        patched.append((namespace, name))
            input_types = node_class.INPUT_TYPES()
        finally:
            folder_paths.get_filename_list = original
            for namespace, name in patched:
                namespace[name] = original

    return input_types, markers


def _combo_options(spec) -> Optional[list]:
    """Option list of a combo input spec, in either the list or COMBO form."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], (list, tuple)):
        return list(spec[0])
    if spec[0] == 'COMBO' and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get('options')
        if isinstance(options, (list, tuple)):
            return list(options)
    return None


def _derive_model_fields(node_class) -> Optional[Dict[str, Any]]:
    """Map a node class's widget indices to the model categories they load."""
    input_types, markers = _trace_input_types(node_class)

    all_categories: List[str] = []
    by_widget_index: Dict[int, List[str]] = {}
    widget_index = 0

    for section in ('required', 'optional'):
        inputs = input_types.get(section) or {}
        for name, spec in inputs.items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            extra = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            options = _combo_options(spec)
            is_widget = options is not None or (
                isinstance(spec[0], str) and spec[0] in _WIDGET_INPUT_TYPES)
            if not is_widget or extra.get('forceInput'):
                continue  # a link input - doesn't occupy a widget slot

            categories = _categories_for_options(options, markers) if options else []
            if categories:
                by_widget_index[widget_index] = categories
                for category in categories:
                    if category not in all_categories:
                        all_categories.append(category)

            widget_index += 1
            if extra.get('control_after_generate') or (
                    spec[0] == 'INT' and name in _SEED_INPUT_NAMES):
                widget_index += 1

    if not all_categories:
        return None
    return {
        'all': all_categories,
        'by_widget_index': by_widget_index,
        'widget_count': widget_index,
    }


def _categories_for_options(options: list, markers: Dict[str, str]) -> List[str]:
    """Categories a combo loads from: traced markers first, else folder matching."""
    categories = []
    for option in options:
        if isinstance(option, str):
            for match in _SENTINEL_PATTERN.finditer(option):
                category = markers.get(match.group(0))
                if category and category not in categories:
                    categories.append(category)
    if categories:
        return categories

    # Combo filled without get_filename_list (os.listdir, cached lists, ...):
    # match its real option list back to a folder
    real_options = [opt for opt in options if isinstance(opt, str)]
    if real_options and len(real_options) == len(options) and any(
            is_model_filename(opt) for opt in real_options):
        category = _find_category_for_options(real_options)
        if category:
            return [category]
    return []


def _find_category_for_options(options) -> Optional[str]:
    """Find the folder category whose file list contains all the options."""
    if folder_paths is None:
        return None

    option_set = set(options)
    best = None
    for category in folder_paths.folder_names_and_paths.keys():
        if category in ('custom_nodes', 'configs'):
            continue
        try:
            files = set(folder_paths.get_filename_list(category))
        except Exception:
            continue
        if option_set <= files:
            # Prefer the tightest-fitting category when several qualify
            if best is None or len(files) < best[1]:
                best = (category, len(files))

    return best[0] if best else None


def try_resolve_model_path(value: str, categories: List[str] = None) -> Optional[tuple[str, str]]:
    """
    Try to resolve a model path using folder_paths.
    
    Args:
        value: The model filename/path to resolve
        categories: Optional list of categories to try (if None, tries all)
        
    Returns:
        Tuple of (category, full_path) if found, None otherwise
    """
    if not isinstance(value, str) or not value.strip():
        return None
    
    # Remove any path separators that might indicate an absolute path prefix
    # Workflows should store relative paths, but handle both cases
    filename = value.strip()

    # Workflows authored on another OS may use the opposite path separator;
    # folder_paths won't resolve those, so also try separator-swapped variants
    variants = [filename]
    if os.sep == '\\':
        swapped = filename.replace('/', '\\')
    else:
        swapped = filename.replace('\\', '/')
    if swapped != filename:
        variants.append(swapped)
    
    # Ensure folder_paths is available
    global folder_paths
    if folder_paths is None:
        try:
            import folder_paths as fp
            folder_paths = fp
        except ImportError:
            logging.error("Model Linker: folder_paths not available")
            return None
    
    # If categories not provided, try all categories
    if categories is None:
        categories = list(folder_paths.folder_names_and_paths.keys())
    
    # Skip non-model categories
    skip_categories = {'custom_nodes', 'configs'}
    categories = [c for c in categories if c not in skip_categories]
    
    for category in categories:
        for variant in variants:
            try:
                full_path = folder_paths.get_full_path(category, variant)
                if full_path and os.path.exists(full_path):
                    return (category, full_path)
            except Exception:
                continue

    return None


def get_node_model_info(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract model references from a single node.
    
    This scans all widgets_values entries and tries to identify which ones
    are model file references by attempting to resolve them.
    
    Args:
        node: Node dictionary from workflow JSON
        
    Returns:
        List of model reference dictionaries:
        {
            'node_id': node id,
            'node_type': node type,
            'widget_index': index in widgets_values,
            'original_path': original path from workflow,
            'category': model category (if found),
            'exists': True if model exists
        }
    """
    model_refs = []
    node_id = node.get('id')
    node_type = node.get('type', '')
    widgets_values = node.get('widgets_values', [])
    
    if not widgets_values:
        return model_refs
    
    # Get category hints for this node type: hardcoded table first, then
    # INPUT_TYPES introspection for custom loader nodes (issue #5)
    category_hint = NODE_TYPE_TO_CATEGORY_HINTS.get(node_type)
    field_info = None
    if category_hint:
        node_categories = [category_hint]
    else:
        field_info = get_node_model_field_info(node_type)
        node_categories = field_info['all'] if field_info else None

    # Per-widget categories only line up when the saved values match the
    # widgets we counted; otherwise use the node-wide list
    by_widget_index = None
    if field_info and len(widgets_values) == field_info['widget_count']:
        by_widget_index = field_info['by_widget_index']

    # For each widget value, check if it looks like a model file
    for idx, value in enumerate(widgets_values):
        if not is_model_filename(value):
            continue

        expected_categories = node_categories
        if by_widget_index and idx in by_widget_index:
            expected_categories = by_widget_index[idx]
        categories_to_try = expected_categories  # None -> try all categories

        # Try to resolve the model path
        resolved = try_resolve_model_path(value, categories_to_try)

        if resolved:
            category, full_path = resolved
            exists = os.path.exists(full_path)
        else:
            # If we can't resolve it, check if it at least looks like a model filename
            # This might be a missing model or a custom node's model
            category = (expected_categories[0] if expected_categories else None) or 'unknown'
            full_path = None
            exists = False

        model_refs.append({
            'node_id': node_id,
            'node_type': node_type,
            'widget_index': idx,
            'original_path': value,
            'category': category,
            'expected_categories': expected_categories,
            'full_path': full_path,
            'exists': exists
        })

    return model_refs


def analyze_workflow_models(workflow_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all model references from a workflow, including nested subgraphs.
    
    Args:
        workflow_json: Complete workflow JSON dictionary
        
    Returns:
        List of model reference dictionaries (same format as get_node_model_info)
        Each dict includes 'subgraph_id' if the model is in a subgraph
    """
    all_model_refs = []
    
    # Get subgraph definitions first to check if node types are subgraph UUIDs
    definitions = workflow_json.get('definitions', {})
    subgraphs = definitions.get('subgraphs', [])
    subgraph_lookup = {sg.get('id'): sg.get('name', sg.get('id')) for sg in subgraphs}
    
    # Analyze top-level nodes
    nodes = workflow_json.get('nodes', [])
    for node in nodes:
        try:
            model_refs = get_node_model_info(node)
            node_type = node.get('type', '')
            
            # Check if node type is a subgraph UUID
            subgraph_name = None
            subgraph_id = None
            if node_type in subgraph_lookup:
                subgraph_name = subgraph_lookup[node_type]
                subgraph_id = node_type
            
            # Mark with subgraph info if it's a subgraph node
            # For top-level subgraph instance nodes, subgraph_path is None
            # This distinguishes them from nodes within subgraph definitions
            for ref in model_refs:
                ref['subgraph_id'] = subgraph_id
                ref['subgraph_name'] = subgraph_name
                ref['subgraph_path'] = None  # Top-level, not in definitions.subgraphs
                ref['is_top_level'] = True  # Flag to indicate this is a top-level node
            all_model_refs.extend(model_refs)
        except Exception as e:
            logging.warning(f"Error analyzing node {node.get('id', 'unknown')}: {e}")
            continue
    
    # Recursively analyze subgraphs (definitions already loaded above)
    if not subgraphs:  # Re-get if not loaded above
        subgraphs = definitions.get('subgraphs', [])
    
    for subgraph in subgraphs:
        subgraph_id = subgraph.get('id')
        subgraph_name = subgraph.get('name', subgraph_id)
        subgraph_nodes = subgraph.get('nodes', [])
        
        logging.debug(f"Analyzing subgraph: {subgraph_name} (ID: {subgraph_id}) with {len(subgraph_nodes)} nodes")
        
        for node in subgraph_nodes:
            try:
                model_refs = get_node_model_info(node)
                # Mark as belonging to this subgraph definition
                for ref in model_refs:
                    ref['subgraph_id'] = subgraph_id
                    ref['subgraph_name'] = subgraph_name
                    ref['subgraph_path'] = ['definitions', 'subgraphs', subgraph_id, 'nodes']
                    ref['is_top_level'] = False  # This is inside a subgraph definition
                all_model_refs.extend(model_refs)
            except Exception as e:
                logging.warning(f"Error analyzing subgraph node {node.get('id', 'unknown')}: {e}")
                continue
    
    return all_model_refs


def group_models_by_file(
    workflow_models: List[Dict[str, Any]],
    exists_filter: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Group model references by file so each model appears once even when
    referenced by multiple nodes.

    Args:
        workflow_models: List of model references from analyze_workflow_models
        exists_filter: False -> only missing refs, True -> only resolved refs,
                       None -> all refs

    Returns:
        List of grouped model references (deduplicated by original_path).
        Each entry has 'all_node_refs' containing all node references for that model.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for model_ref in workflow_models:
        if exists_filter is not None and model_ref.get('exists', False) != exists_filter:
            continue

        filename = model_ref.get('original_path', '')

        if filename not in grouped:
            # First occurrence - use this as the primary entry
            grouped[filename] = {
                **model_ref,
                'all_node_refs': [model_ref.copy()]  # Track all nodes using this model
            }
        else:
            # Duplicate - just add to the node refs list
            grouped[filename]['all_node_refs'].append(model_ref.copy())

    return list(grouped.values())


def identify_missing_models(
    workflow_models: List[Dict[str, Any]],
    available_models: List[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Identify which models from the workflow are missing.
    Deduplicates by filename - same model file only appears once even if
    referenced by multiple nodes.

    Args:
        workflow_models: List of model references from analyze_workflow_models
        available_models: Optional list of available models (if None, checks via folder_paths)

    Returns:
        List of missing model references (deduplicated by filename).
        Each entry has 'all_node_refs' containing all node references for that model.
    """
    return group_models_by_file(workflow_models, exists_filter=False)

