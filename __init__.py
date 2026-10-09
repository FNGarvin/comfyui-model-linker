"""
@author: Model Linker Team
@title: ComfyUI Model Linker
@nickname: Model Linker
@version: 1.1.0
@description: Extension for relinking missing models and downloading from HuggingFace/CivitAI
"""

import logging

# Web directory for JavaScript interface
WEB_DIRECTORY = "./web"

# Empty NODE_CLASS_MAPPINGS - we don't provide custom nodes, only web extension
# This prevents ComfyUI from showing "IMPORT FAILED" message
NODE_CLASS_MAPPINGS = {}

__all__ = ["WEB_DIRECTORY"]


class ModelLinkerExtension:
    """Main extension class for Model Linker."""
    
    def __init__(self):
        self.routes_setup = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize the extension and set up API routes."""
        try:
            self.setup_routes()
            self.logger.info("Model Linker: Extension initialized successfully")
        except Exception as e:
            self.logger.error(f"Model Linker: Extension initialization failed: {e}", exc_info=True)
    
    def setup_routes(self):
        """Register API routes for the Model Linker extension."""
        if self.routes_setup:
            return  # Already set up
        
        try:
            from aiohttp import web
            
            # Try to get routes from PromptServer
            try:
                from server import PromptServer
                if not hasattr(PromptServer, 'instance') or PromptServer.instance is None:
                    self.logger.debug("Model Linker: PromptServer not available yet")
                    return False
                
                routes = PromptServer.instance.routes
            except (ImportError, AttributeError) as e:
                self.logger.debug(f"Model Linker: Could not access PromptServer: {e}")
                return False
            
            # Import linker modules
            try:
                from .core.linker import analyze_and_find_matches, apply_resolution
                from .core.scanner import get_model_files
                from .core.matcher import find_matches
            except ImportError as e:
                self.logger.error(f"Model Linker: Could not import core modules: {e}")
                return False
            
            from .core.settings import (
                public_view, save_settings, check_token, get_token, auth_headers_for_url
            )

            # Import download modules
            try:
                from .core.source_chain import find_download_source, find_by_hash
                from .core.sources import hf_index
                from .core.downloader import (
                    start_background_download, get_progress, get_all_progress,
                    cancel_download, get_download_directory
                )
                from .core.sources.popular import get_popular_model_url, search_popular_models
                from .core.sources.model_list import search_model_list, search_model_list_multiple
                from .core.sources.huggingface import search_huggingface_for_file
                from .core.sources.civitai import search_civitai_for_file
                download_available = True
            except ImportError as e:
                self.logger.warning(f"Model Linker: Download features not available: {e}")
                download_available = False
            
            import asyncio

            def attach_download_sources(missing_models):
                """Attach a download_source to each missing model without a 100% local match."""
                for missing in missing_models:
                    matches = missing.get('matches', [])
                    if any(m.get('confidence', 0) == 100 for m in matches):
                        continue
                    source = find_download_source(
                        missing.get('original_path', ''),
                        category=missing.get('category', ''),
                        workflow_url=missing.get('workflow_url', ''),
                        workflow_directory=missing.get('workflow_directory', ''),
                        sha256=missing.get('workflow_sha256', ''),
                    )
                    if source:
                        missing['download_source'] = source

            # ==================== ANALYZE ROUTES ====================
            
            @routes.post("/model_linker/analyze")
            async def analyze_workflow(request):
                """Analyze workflow and return missing models with matches."""
                try:
                    data = await request.json()
                    workflow_json = data.get('workflow')
                    
                    if not workflow_json:
                        return web.json_response(
                            {'error': 'Workflow JSON is required'},
                            status=400
                        )
                    
                    # Analyze and find matches, then look up download sources
                    # for models without a 100% local match. Both block on
                    # disk/network, so keep them off the event loop.
                    include_resolved = bool(data.get('include_resolved', False))
                    result = await asyncio.to_thread(
                        analyze_and_find_matches, workflow_json, include_resolved=include_resolved)

                    if download_available:
                        await asyncio.to_thread(attach_download_sources, result.get('missing_models', []))

                    return web.json_response(result)
                except Exception as e:
                    self.logger.error(f"Model Linker analyze error: {e}", exc_info=True)
                    return web.json_response(
                        {'error': str(e)},
                        status=500
                    )
            
            @routes.post("/model_linker/resolve")
            async def resolve_models(request):
                """Apply model resolution and return updated workflow."""
                try:
                    data = await request.json()
                    workflow_json = data.get('workflow')
                    resolutions = data.get('resolutions', [])
                    
                    if not workflow_json:
                        return web.json_response(
                            {'error': 'Workflow JSON is required'},
                            status=400
                        )
                    
                    if not resolutions:
                        return web.json_response(
                            {'error': 'Resolutions array is required'},
                            status=400
                        )
                    
                    # Apply resolutions
                    updated_workflow = apply_resolution(workflow_json, resolutions)
                    
                    return web.json_response({
                        'workflow': updated_workflow,
                        'success': True
                    })
                except Exception as e:
                    self.logger.error(f"Model Linker resolve error: {e}", exc_info=True)
                    return web.json_response(
                        {'error': str(e), 'success': False},
                        status=500
                    )
            
            @routes.get("/model_linker/models")
            async def get_models(request):
                """Get list of available models.

                Optional query params (used by the All models swap picker):
                - category: comma-separated category names to filter by
                - current: a filename; results are sorted by similarity to it
                """
                try:
                    models = get_model_files()

                    category_param = request.query.get('category', '')
                    if category_param:
                        wanted = {c.strip() for c in category_param.split(',') if c.strip()}
                        models = [m for m in models if m.get('category') in wanted]

                    current = request.query.get('current', '')
                    if current and models:
                        ranked = find_matches(current, models, threshold=0.0, max_results=len(models))
                        models = [
                            {**match['model'], 'confidence': match['confidence']}
                            for match in ranked
                        ]

                    return web.json_response(models)
                except Exception as e:
                    self.logger.error(f"Model Linker get_models error: {e}", exc_info=True)
                    return web.json_response(
                        {'error': str(e)},
                        status=500
                    )
            
            # ==================== SETTINGS ROUTES ====================

            @routes.get("/model_linker/settings")
            async def get_settings_route(request):
                """Token status for the settings panel - never the tokens themselves."""
                return web.json_response(public_view())

            @routes.post("/model_linker/settings")
            async def save_settings_route(request):
                """Save or clear tokens: {hf_token?, civitai_key?}; '' clears."""
                try:
                    data = await request.json()
                    updates = {}
                    for key in ('hf_token', 'civitai_key'):
                        if key in data:
                            value = data.get(key)
                            if value is not None and not isinstance(value, str):
                                return web.json_response({'error': f'{key} must be a string'}, status=400)
                            updates[key] = (value or '').strip()
                    save_settings(updates)
                    return web.json_response(public_view())
                except Exception as e:
                    self.logger.error(f"Model Linker settings error: {e}", exc_info=True)
                    return web.json_response({'error': str(e)}, status=500)

            @routes.post("/model_linker/settings/check/{provider}")
            async def check_settings_route(request):
                """Validate a stored token against the provider's API."""
                import asyncio
                provider = request.match_info['provider']
                result = await asyncio.to_thread(check_token, provider)
                return web.json_response(result)

            # ==================== DOWNLOAD ROUTES ====================
            
            if download_available:
                
                @routes.post("/model_linker/search")
                async def search_sources(request):
                    """Search for model download sources."""
                    try:
                        data = await request.json()
                        filename = data.get('filename', '')
                        category = data.get('category', '')
                        
                        if not filename:
                            return web.json_response(
                                {'error': 'Filename is required'},
                                status=400
                            )
                        
                        results = {
                            'popular': None,
                            'model_list': None,
                            'huggingface': None,
                            'civitai': None,
                            'found': False
                        }
                        
                        # 1. Check popular models first (curated database)
                        popular_info = get_popular_model_url(filename)
                        if popular_info:
                            results['popular'] = {
                                'source': 'popular',
                                'filename': filename,
                                **popular_info
                            }
                            results['found'] = True
                        
                        # 2. Search model-list.json (ComfyUI Manager database with fuzzy matching)
                        if not results['found']:
                            model_list_result = search_model_list(filename)
                            if model_list_result:
                                results['model_list'] = model_list_result
                                results['found'] = True
                        
                        # 3. Bundled Comfy-Org/Kijai index, then hash lookups
                        if not results['found']:
                            index_result = hf_index.lookup_by_filename(filename)
                            if index_result:
                                results['huggingface'] = {**index_result, 'repo_id': index_result['repo_id']}
                                results['found'] = True
                        sha256 = data.get('sha256', '')
                        if not results['found'] and sha256:
                            hash_result = await asyncio.to_thread(find_by_hash, sha256, filename)
                            if hash_result:
                                key = 'civitai' if hash_result.get('source') == 'civitai' else 'huggingface'
                                results[key] = {**hash_result, 'download_url': hash_result['url']}
                                results['found'] = True

                        # 4. Search HuggingFace for exact file match
                        if not results['found']:
                            hf_result = search_huggingface_for_file(filename, token=get_token('huggingface'))
                            if hf_result:
                                results['huggingface'] = hf_result
                                results['found'] = True
                        
                        # 5. Search CivitAI for exact file match
                        if not results['found']:
                            civitai_result = search_civitai_for_file(filename, api_key=get_token('civitai'))
                            if civitai_result:
                                results['civitai'] = civitai_result
                                results['found'] = True
                        
                        return web.json_response(results)
                        
                    except Exception as e:
                        self.logger.error(f"Model Linker search error: {e}", exc_info=True)
                        return web.json_response(
                            {'error': str(e)},
                            status=500
                        )
                
                @routes.post("/model_linker/download")
                async def download_model(request):
                    """Start downloading a model."""
                    try:
                        data = await request.json()
                        url = data.get('url', '')
                        filename = data.get('filename', '')
                        category = data.get('category', 'checkpoints')
                        subfolder = data.get('subfolder', '')
                        
                        if not url:
                            return web.json_response(
                                {'error': 'URL is required'},
                                status=400
                            )
                        
                        if not filename:
                            # Extract filename from URL
                            from urllib.parse import urlparse, unquote
                            parsed = urlparse(url)
                            filename = unquote(parsed.path.split('/')[-1])
                        
                        if not filename:
                            return web.json_response(
                                {'error': 'Could not determine filename'},
                                status=400
                            )
                        
                        # Tokens come from the server-side settings, never
                        # from the browser; sent as a header (not in the URL)
                        # so they don't end up in logs. requests drops the
                        # header when CivitAI redirects to its file storage.
                        headers = auth_headers_for_url(url)
                        
                        # Start background download
                        download_id = start_background_download(
                            url=url,
                            filename=filename,
                            category=category,
                            headers=headers if headers else None,
                            subfolder=subfolder
                        )
                        
                        return web.json_response({
                            'success': True,
                            'download_id': download_id,
                            'filename': filename,
                            'category': category
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Model Linker download error: {e}", exc_info=True)
                        return web.json_response(
                            {'error': str(e), 'success': False},
                            status=500
                        )
                
                @routes.get("/model_linker/progress/{download_id}")
                async def get_download_progress(request):
                    """Get progress for a specific download."""
                    try:
                        download_id = request.match_info['download_id']
                        progress = get_progress(download_id)
                        
                        if progress:
                            return web.json_response(progress)
                        else:
                            return web.json_response(
                                {'error': 'Download not found'},
                                status=404
                            )
                    except Exception as e:
                        self.logger.error(f"Model Linker progress error: {e}", exc_info=True)
                        return web.json_response(
                            {'error': str(e)},
                            status=500
                        )
                
                @routes.get("/model_linker/progress")
                async def get_all_downloads_progress(request):
                    """Get progress for all downloads."""
                    try:
                        progress = get_all_progress()
                        return web.json_response(progress)
                    except Exception as e:
                        self.logger.error(f"Model Linker progress error: {e}", exc_info=True)
                        return web.json_response(
                            {'error': str(e)},
                            status=500
                        )
                
                @routes.post("/model_linker/cancel/{download_id}")
                async def cancel_download_route(request):
                    """Cancel a download in progress."""
                    try:
                        download_id = request.match_info['download_id']
                        cancel_download(download_id)
                        return web.json_response({'success': True})
                    except Exception as e:
                        self.logger.error(f"Model Linker cancel error: {e}", exc_info=True)
                        return web.json_response(
                            {'error': str(e), 'success': False},
                            status=500
                        )
                
                @routes.get("/model_linker/directories")
                async def get_directories(request):
                    """Get available model directories."""
                    try:
                        categories = [
                            'checkpoints', 'loras', 'vae', 'controlnet', 
                            'clip', 'clip_vision', 'embeddings', 'upscale_models',
                            'diffusion_models', 'text_encoders', 'ipadapter', 'sams'
                        ]
                        
                        directories = {}
                        for cat in categories:
                            path = get_download_directory(cat)
                            if path:
                                directories[cat] = path
                        
                        return web.json_response(directories)
                    except Exception as e:
                        self.logger.error(f"Model Linker directories error: {e}", exc_info=True)
                        return web.json_response(
                            {'error': str(e)},
                            status=500
                        )
            
            self.routes_setup = True
            self.logger.info("Model Linker: API routes registered successfully")
            return True
            
        except ImportError as e:
            self.logger.warning(f"Model Linker: Could not register routes (missing dependency): {e}")
            return False
        except Exception as e:
            self.logger.error(f"Model Linker: Error setting up routes: {e}", exc_info=True)
            return False


# Initialize the extension
try:
    extension = ModelLinkerExtension()
    extension.initialize()
except Exception as e:
    logging.error(f"ComfyUI Model Linker extension initialization failed: {e}", exc_info=True)
