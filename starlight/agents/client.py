import requests
import logging
import hashlib
import time
from typing import List, Dict, Optional, Any, Union
from .config import Config

logger = logging.getLogger(__name__)

class StargateClient:
    def __init__(self, api_url: str = Config.STARGATE_API_URL, api_key: str = Config.STARGATE_API_KEY):
        self.api_key = api_key
        self.base_url = api_url.rstrip("/")
        self.mcp_url = f"{self.base_url}/mcp/call"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "Authorization": f"Bearer {api_key}" # Support both auth styles
        }

    def _api_key_hash(self) -> str:
        """Returns the hex-encoded SHA256 hash of the API key."""
        if not self.api_key:
            return ""
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Union[Dict, List]]:
        """Helper to make HTTP requests with container-optimized error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            # Container-optimized timeouts: shorter for faster failure detection
            response = requests.request(method, url, headers=self.headers, timeout=5, **kwargs)
            
            if response.status_code == 429:
                logger.warning(f"Throttled by Stargate API (429) at {endpoint}. Backing off.")
                time.sleep(1)  # Brief backoff for throttling
                return None
            
            response.raise_for_status()
            
            try:
                return response.json()
            except ValueError:
                # Some endpoints might return empty body on success (e.g. 204)
                return {}
                
        except requests.Timeout:
            logger.warning(f"Request timeout for {method} {endpoint} (container optimization)")
            return None
        except requests.ConnectionError:
            logger.warning(f"Connection error for {method} {endpoint} (container network)")
            return None
        except requests.RequestException as e:
            logger.error(f"Request failed for {method} {endpoint}: {e}")
            if getattr(e, 'response', None) is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def mcp_call(self, tool: str, arguments: Optional[Dict] = None) -> Optional[Any]:
        """Makes an MCP call to Stargate backend with container optimizations."""
        if arguments is None:
            arguments = {}
            
        payload = {
            "tool": tool,
            "arguments": arguments
        }
        
        try:
            # Container-optimized timeout: faster failure for network issues
            response = requests.post(self.mcp_url, json=payload, headers=self.headers, timeout=8)
            
            if response.status_code == 429:
                logger.warning(f"Throttled by Stargate MCP (429) for tool {tool}. Backing off.")
                time.sleep(0.5)  # Brief backoff for throttling
                return None
                
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, dict) and "error" in result:
                logger.error(f"MCP error for {tool}: {result['error']}")
                return None
                
            # Extract the actual result from MCP response format
            if isinstance(result, dict):
                return result.get("result") or result.get("content") or result
            return result
            
        except requests.Timeout:
            logger.warning(f"MCP timeout for {tool} (container optimization)")
            return None
        except requests.ConnectionError:
            logger.warning(f"MCP connection error for {tool} (container network)")
            return None
        except requests.RequestException as e:
            logger.error(f"MCP call failed for {tool}: {e}")
            if getattr(e, 'response', None) is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def bind_wallet(self, wallet_address: str) -> bool:
        """Binds a wallet address to the current API key."""
        payload = {
            "api_key": self.headers.get("X-API-Key"),
            "wallet_address": wallet_address
        }
        # Use /api/auth/login which supports binding if not already bound
        result = self._request("POST", "/api/auth/login", json=payload)
        if result and result.get("valid"):
            logger.info(f"Successfully bound wallet {wallet_address} to API key.")
            return True
        return False

    def get_open_contracts(self) -> List[Dict]:
        """Fetches all open contracts/wishes using MCP."""
        result = self.mcp_call("get_open_contracts", {"status": "pending"})
        if result is None:
            return []
        if isinstance(result, dict):
            return result.get("contracts") or []
        return result if isinstance(result, list) else []

    def get_contract(self, contract_id: str) -> Optional[Dict]:
        """Fetches a specific contract by ID."""
        result = self.mcp_call("get_contract", {"contract_id": contract_id})
        if result is None:
            return None
        if isinstance(result, dict):
            return result.get("contracts", [result])[0] if result.get("contracts") else result
        return None

    def get_proposals(self, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
         """Fetches proposals using MCP, optionally filtered by status. Supports pagination."""
         args = {"limit": limit, "offset": offset}
         if status:
             args["status"] = status
         result = self.mcp_call("list_proposals", args)
         if result is None:
             return []
         if isinstance(result, dict):
             return result.get("proposals") or []
         return result if isinstance(result, list) else []

    def create_proposal(self, proposal_data: Dict) -> Optional[str]:
        """Creates a new proposal using MCP."""
        if "metadata" not in proposal_data:
            proposal_data["metadata"] = {}
        proposal_data["metadata"]["creator_api_key_hash"] = self._api_key_hash()
        
        result = self.mcp_call("create_proposal", proposal_data)
        if result is None or not isinstance(result, dict):
            return None
        # Result might be { "proposal": { ... } } or { "id": "..." }
        p = result.get("proposal")
        if isinstance(p, dict):
            return p.get("id") or p.get("proposal_id")
        return result.get("id") or result.get("proposal_id")

    def update_proposal(self, proposal_id: str, proposal_data: Dict) -> bool:
        """Updates an existing proposal."""
        url = f"/api/smart_contract/proposals/{proposal_id}"
        result = self._request("PATCH", url, json=proposal_data)
        return result is not None

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approves a proposal using MCP."""
        result = self.mcp_call("approve_proposal", {"proposal_id": proposal_id})
        return result is not None

    def get_tasks(self, contract_id: str) -> List[Dict]:
        """Fetches tasks using MCP."""
        result = self.mcp_call("list_tasks", {"contract_id": contract_id})
        if result is None:
            return []
        if isinstance(result, dict):
            return result.get("tasks") or []
        return result if isinstance(result, list) else []

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Fetches detailed status for a single task."""
        # Use mcp tool get_task if available, otherwise fallback to direct API
        result = self.mcp_call("get_task", {"task_id": task_id})
        if result and isinstance(result, dict):
            return result
        
        # Fallback to direct status API
        return self._request("GET", f"/api/smart_contract/tasks/{task_id}/status")

    def claim_task(self, task_id: str, ai_identifier: str) -> Optional[str]:
        """Claims a task using MCP."""
        args = {
            "task_id": task_id,
            "ai_identifier": ai_identifier
        }
        result = self.mcp_call("claim_task", args)
        if result is None:
            return None
            
        if isinstance(result, dict):
             # Response wrapped: { "claim": { "claim_id": "..." } }
             claim = result.get("claim", {})
             if isinstance(claim, dict):
                 return claim.get("claim_id")
             return result.get("claim_id")
        return None

    def submit_work(self, claim_id: str, deliverables: Dict) -> bool:
        """Submits work using MCP."""
        args = {
            "claim_id": claim_id,
            "deliverables": deliverables
        }
        result = self.mcp_call("submit_work", args)
        return result is not None

    def get_submissions(self, contract_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """Fetches submissions, optionally filtered by contract_id or status."""
        params = {}
        if contract_id:
            params["contract_id"] = contract_id
        if status:
            params["status"] = status
        data = self._request("GET", "/api/smart_contract/submissions", params=params)
        if data is None:
            return []
        
        raw_submissions = []
        if isinstance(data, dict):
            raw_submissions = data.get("submissions", [])
        elif isinstance(data, list):
            raw_submissions = data
            
        # If the backend returns a list of IDs (strings), we need to fetch full objects
        full_submissions = []
        for item in raw_submissions:
            if isinstance(item, str):
                full_sub = self.get_submission(item)
                if full_sub:
                    full_submissions.append(full_sub)
            elif isinstance(item, dict):
                full_submissions.append(item)
                
        return full_submissions

    def get_submission(self, submission_id: str) -> Optional[Dict]:
        """Fetches a single submission by ID."""
        data = self._request("GET", f"/api/smart_contract/submissions/{submission_id}")
        if data and isinstance(data, dict):
            # Handle wrapped response { "submission": { ... } }
            return data.get("submission") or data
        return None

    def review_submission(self, submission_id: str, action: str, notes: str = "") -> bool:
        """Reviews a submission (approve/reject)."""
        url = f"/api/smart_contract/submissions/{submission_id}/review"
        payload = {
            "action": action, # 'approve', 'reject', or 'review'
            "notes": notes
        }
        result = self._request("POST", url, json=payload)
        return result is not None
