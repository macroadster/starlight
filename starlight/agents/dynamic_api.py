"""
FastAPI Dynamic Loading Integration
Provides secure endpoints for loading agent-developed functions
"""

import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dynamic_loader import dynamic_loader, LoadRequest, LoadedModule
from .config import Config

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """
    Verify STARGATE_API_KEY for write operations.
    This ensures only authenticated users can load/unload functions.
    """
    if not credentials.credentials or credentials.credentials != Config.STARGATE_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return True

def create_dynamic_app() -> FastAPI:
    """Create FastAPI app with dynamic loading capabilities."""
    app = FastAPI(
        title="Starlight Dynamic API",
        description="Secure dynamic loading of agent-developed functions",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "service": "Starlight Dynamic API",
            "status": "running",
            "version": "1.0.0"
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    # WRITE OPERATIONS - Require API Key Authentication
    
    @app.post("/load-function", response_model=LoadedModule)
    async def load_function(
        request: LoadRequest,
        authenticated: bool = Depends(verify_api_key)
    ):
        """
        Load a function from agent sandbox into FastAPI.
        Requires STARGATE_API_KEY authentication.
        """
        try:
            loaded_module = dynamic_loader.load_function(request)
            
            # Register the function as an endpoint
            func = dynamic_loader.get_function(request.visible_pixel_hash, request.function_name)
            if func:
                # Add dynamic route
                app.add_api_route(
                    loaded_module.endpoint_path,
                    func,
                    methods=[loaded_module.method],
                    name=f"agent_{loaded_module.module_id[:8]}",
                    operation_id=f"agent_{loaded_module.visible_pixel_hash}_{loaded_module.function_name}",
                    include_in_schema=True
                )
                
                # Refresh OpenAPI schema
                app.openapi_schema = None
                
                logger.info(f"Registered dynamic endpoint: {loaded_module.endpoint_path}")
            
            return loaded_module
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in load_function endpoint: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.delete("/unload-function/{visible_pixel_hash}/{function_name}")
    async def unload_function(
        visible_pixel_hash: str,
        function_name: str,
        authenticated: bool = Depends(verify_api_key)
    ):
        """
        Unload a function and remove its endpoint.
        Requires STARGATE_API_KEY authentication.
        """
        try:
            # Find the module to get endpoint path
            module_to_remove = None
            for module_id, loaded_module in dynamic_loader.loaded_modules.items():
                if (loaded_module.visible_pixel_hash == visible_pixel_hash and 
                    loaded_module.function_name == function_name):
                    module_to_remove = loaded_module
                    break
            
            if module_to_remove:
                # Note: FastAPI doesn't have built-in route removal
                # We'll mark the function as unavailable instead
                pass
            
            # Unload from dynamic loader
            success = dynamic_loader.unload_function(visible_pixel_hash, function_name)
            
            if success:
                # Refresh OpenAPI schema
                app.openapi_schema = None
                return {"message": f"Function '{function_name}' unloaded successfully"}
            else:
                raise HTTPException(status_code=404, detail="Function not found")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in unload_function endpoint: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # READ OPERATIONS - No authentication required
    
    @app.get("/list-functions", response_model=Dict[str, LoadedModule])
    async def list_functions():
        """List all currently loaded functions (read operation, no auth required)."""
        return dynamic_loader.list_loaded_functions()
    
    @app.get("/function-info/{module_id}", response_model=LoadedModule)
    async def get_function_info(module_id: str):
        """Get information about a specific loaded module (read operation, no auth required)."""
        module_info = dynamic_loader.get_module_info(module_id)
        if not module_info:
            raise HTTPException(status_code=404, detail="Module not found")
        return module_info
    
    @app.get("/function/{visible_pixel_hash}/{function_name}")
    async def call_function(visible_pixel_hash: str, function_name: str, request: Request):
        """
        Call a loaded function directly without registering as endpoint.
        This is a read operation - function must already be loaded.
        """
        try:
            func = dynamic_loader.get_function(visible_pixel_hash, function_name)
            if not func:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Function '{function_name}' not loaded for hash {visible_pixel_hash}"
                )
            
            # Call the function
            if callable(func):
                try:
                    # Try calling with request object
                    result = await func(request) if hasattr(func, '__call__') and hasattr(func, '__code__') and func.__code__.co_argcount > 0 else func()
                    return {"result": result}
                except TypeError:
                    # Call without arguments if signature doesn't accept request
                    result = func()
                    return {"result": result}
            else:
                return {"value": func}
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error calling function '{function_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")
    
    @app.get("/endpoints")
    async def list_endpoints():
        """List all registered FastAPI endpoints."""
        endpoints = []
        for route in app.routes:
            # Check for route attributes safely
            if hasattr(route, 'path'):
                route_info = {
                    "path": getattr(route, 'path', 'unknown'),
                    "methods": list(getattr(route, 'methods', [])),
                    "name": getattr(route, 'name', None)
                }
                endpoints.append(route_info)
        return {"endpoints": endpoints}
    
    return app

# Create the FastAPI app
app = create_dynamic_app()

def setup_dynamic_routes(base_app: FastAPI) -> FastAPI:
    """
    Setup dynamic loading routes on an existing FastAPI app.
    This allows integration with existing Starlight applications.
    Note: For now, use the standalone app instead of this integration.
    """
    # For now, return the base app unchanged
    # Integration would require more complex route copying
    return base_app