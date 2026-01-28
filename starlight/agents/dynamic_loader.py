"""
Dynamic Module Loader for Starlight Agent Integration
Handles secure loading of agent-developed functions into FastAPI
"""

import os
import importlib.util
import ast
import logging
from typing import Dict, Any, Optional, Callable
from fastapi import HTTPException
from pydantic import BaseModel
import uuid
import time

from .config import Config

logger = logging.getLogger(__name__)

class LoadRequest(BaseModel):
    visible_pixel_hash: str
    function_name: str
    module_name: Optional[str] = None
    endpoint_path: Optional[str] = None
    method: str = "GET"

class LoadedModule(BaseModel):
    module_id: str
    visible_pixel_hash: str
    function_name: str
    endpoint_path: str
    method: str
    loaded_at: float
    module_path: str

class DynamicLoader:
    """
    Secure dynamic module loader for agent-developed code.
    Integrates with FastAPI for runtime endpoint registration.
    """
    
    def __init__(self):
        self.loaded_modules: Dict[str, LoadedModule] = {}
        self.active_functions: Dict[str, Callable] = {}
        
    def _validate_ast(self, module_path: str) -> bool:
        """
        Validate Python AST for security before loading.
        Prevents dangerous operations like file system access, network calls, etc.
        """
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Security check - walk AST for dangerous operations
            dangerous_nodes = []
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess', 'socket', 'requests', 'urllib']:
                            dangerous_nodes.append(f"Import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ['os', 'subprocess', 'socket', 'requests', 'urllib']:
                        dangerous_nodes.append(f"From Import: {node.module}")
                
                # Check for file operations
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['open', 'exec', 'eval', 'compile']:
                            dangerous_nodes.append(f"Function call: {node.func.attr}")
                    
                    elif isinstance(node.func, ast.Name):
                        if node.func.id in ['open', 'exec', 'eval', 'compile']:
                            dangerous_nodes.append(f"Function call: {node.func.id}")
            
            if dangerous_nodes:
                logger.warning(f"Security violations in {module_path}: {dangerous_nodes}")
                return False
                
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {module_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating {module_path}: {e}")
            return False
    
    def _get_sandbox_path(self, visible_pixel_hash: str) -> str:
        """Get the sandbox directory path for a given visible_pixel_hash."""
        return os.path.join(Config.UPLOADS_DIR, "results", visible_pixel_hash)
    
    def _find_module_file(self, sandbox_dir: str, function_name: str, module_name: Optional[str] = None) -> Optional[str]:
        """
        Find the Python file containing the function in the sandbox directory.
        """
        if module_name:
            # Look for specific module file
            module_path = os.path.join(sandbox_dir, f"{module_name}.py")
            if os.path.exists(module_path):
                return module_path
        else:
            # Search for .py files in sandbox
            for filename in os.listdir(sandbox_dir):
                if filename.endswith('.py'):
                    filepath = os.path.join(sandbox_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if f"def {function_name}(" in content:
                                return filepath
                    except Exception:
                        continue
        
        return None
    
    def load_function(self, request: LoadRequest) -> LoadedModule:
        """
        Load a function from agent sandbox and prepare it for FastAPI registration.
        """
        try:
            # Get sandbox directory
            sandbox_dir = self._get_sandbox_path(request.visible_pixel_hash)
            if not os.path.exists(sandbox_dir):
                raise HTTPException(status_code=404, detail=f"Sandbox directory not found for hash: {request.visible_pixel_hash}")
            
            # Find module file
            module_path = self._find_module_file(sandbox_dir, request.function_name, request.module_name)
            if not module_path:
                raise HTTPException(status_code=404, detail=f"Function '{request.function_name}' not found in sandbox")
            
            # Security validation
            if not self._validate_ast(module_path):
                raise HTTPException(status_code=403, detail="Module contains unsafe code")
            
            # Load module
            module_id = str(uuid.uuid4())
            spec = importlib.util.spec_from_file_location(f"agent_module_{module_id}", module_path)
            if spec is None or spec.loader is None:
                raise HTTPException(status_code=500, detail="Failed to create module spec")
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute module to get functions
            spec.loader.exec_module(module)
            
            # Get the function
            if not hasattr(module, request.function_name):
                raise HTTPException(status_code=404, detail=f"Function '{request.function_name}' not found in module")
            
            func = getattr(module, request.function_name)
            
            # Determine endpoint path
            endpoint_path = request.endpoint_path or f"/agent/{request.visible_pixel_hash}/{request.function_name}"
            
            # Store loaded module info
            loaded_module = LoadedModule(
                module_id=module_id,
                visible_pixel_hash=request.visible_pixel_hash,
                function_name=request.function_name,
                endpoint_path=endpoint_path,
                method=request.method,
                loaded_at=time.time(),
                module_path=module_path
            )
            
            self.loaded_modules[module_id] = loaded_module
            self.active_functions[f"{request.visible_pixel_hash}:{request.function_name}"] = func
            
            logger.info(f"Loaded function '{request.function_name}' from {module_path} as endpoint {endpoint_path}")
            
            return loaded_module
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading function: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load function: {str(e)}")
    
    def get_function(self, visible_pixel_hash: str, function_name: str) -> Optional[Callable]:
        """Get a loaded function by visible_pixel_hash and function name."""
        key = f"{visible_pixel_hash}:{function_name}"
        return self.active_functions.get(key)
    
    def unload_function(self, visible_pixel_hash: str, function_name: str) -> bool:
        """Unload a function and clean up resources."""
        key = f"{visible_pixel_hash}:{function_name}"
        
        if key in self.active_functions:
            # Find and remove from loaded_modules
            to_remove = []
            for module_id, loaded_module in self.loaded_modules.items():
                if (loaded_module.visible_pixel_hash == visible_pixel_hash and 
                    loaded_module.function_name == function_name):
                    to_remove.append(module_id)
            
            for module_id in to_remove:
                del self.loaded_modules[module_id]
            
            del self.active_functions[key]
            
            logger.info(f"Unloaded function '{function_name}' for hash {visible_pixel_hash}")
            return True
        
        return False
    
    def list_loaded_functions(self) -> Dict[str, LoadedModule]:
        """List all currently loaded functions."""
        return self.loaded_modules.copy()
    
    def get_module_info(self, module_id: str) -> Optional[LoadedModule]:
        """Get information about a loaded module."""
        return self.loaded_modules.get(module_id)

# Global instance
dynamic_loader = DynamicLoader()