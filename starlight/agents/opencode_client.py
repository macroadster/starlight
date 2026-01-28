import logging
import requests
import json
from typing import Dict, Optional, Any
from .config import Config

logger = logging.getLogger(__name__)

class OpenCodeMCPClient:
    """
    Direct MCP client for OpenCode operations.
    More efficient than subprocess calls as it reuses existing MCP connections.
    """
    
    def __init__(self):
        # Use the same MCP endpoint as StargateClient but for OpenCode tools
        self.api_key = Config.STARGATE_API_KEY
        self.base_url = Config.STARGATE_API_URL.rstrip("/")
        
        # Try to detect OpenCode MCP endpoint - could be same as stargate or separate
        self.mcp_url = f"{self.base_url}/mcp/call"  # Default to stargate's MCP
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Test if we can reach OpenCode via MCP
        self._available = self._test_connection()
        
    def _test_connection(self) -> bool:
        """Test if OpenCode MCP tools are available."""
        try:
            # Direct test without using self.run() to avoid circular dependency
            payload = {
                "tool": "opencode_run",
                "arguments": {
                    "prompt": "hello",
                    "timeout": 5
                }
            }
            
            response = requests.post(self.mcp_url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("result") is not None:
                    logger.info("OpenCode MCP client connected successfully")
                    return True
        except Exception as e:
            logger.warning(f"OpenCode MCP not available: {e}")
        return False
    
    def is_available(self) -> bool:
        """Check if OpenCode MCP client is available."""
        return self._available
    
    def run(self, prompt: str, timeout: int = 300) -> Optional[str]:
        """
        Execute OpenCode run command via MCP.
        Equivalent to: subprocess.run(["opencode", "run", prompt])
        """
        if not self._available:
            return None
            
        payload = {
            "tool": "opencode_run",
            "arguments": {
                "prompt": prompt,
                "timeout": timeout
            }
        }
        
        try:
            response = requests.post(self.mcp_url, headers=self.headers, json=payload, timeout=timeout+10)
            
            if response.status_code == 429:
                logger.warning("Throttled by OpenCode MCP (429)")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                error = data.get("error", "Unknown error")
                # If tool not found, mark as unavailable to avoid repeated attempts
                if "not found" in str(error).lower():
                    self._available = False
                    logger.warning(f"OpenCode MCP tool 'opencode_run' not found, marking as unavailable: {error}")
                else:
                    logger.error(f"OpenCode MCP failed: {error}")
                return None
                
            return data.get("result")
            
        except requests.RequestException as e:
            logger.error(f"OpenCode MCP request failed: {e}")
            # Don't mark as unavailable for network errors, might be temporary
            return None
        except Exception as e:
            logger.error(f"OpenCode MCP error: {e}")
            return None
    
    def run_with_context(self, prompt: str, context: Optional[Dict] = None, timeout: int = 300) -> Optional[str]:
        """
        Execute OpenCode with additional context.
        This can maintain conversation state and provide better results.
        """
        if not self._available:
            return self.run(prompt, timeout)  # Fallback to basic run
            
        payload = {
            "tool": "opencode_run",
            "arguments": {
                "prompt": prompt,
                "context": context or {},
                "timeout": timeout
            }
        }
        
        try:
            response = requests.post(self.mcp_url, headers=self.headers, json=payload, timeout=timeout+10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                error = data.get("error", "Unknown error")
                logger.error(f"OpenCode MCP with context failed: {error}")
                return None
                
            return data.get("result")
            
        except Exception as e:
            logger.error(f"OpenCode MCP with context error: {e}")
            # Fallback to basic run
            return self.run(prompt, timeout)