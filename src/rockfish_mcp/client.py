import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class RockfishHTTPClient:
    """HTTP/REST client for interacting with the Rockfish API.

    This client handles operations not easily supported by the Rockfish SDK,
    such as database management, worker set management, and certain dataset/model operations.
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.rockfish.ai", organization_id=None, project_id=None):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        if organization_id:
            self.headers["X-Organization-ID"] = organization_id

        if project_id:
            self.headers["X-Project-ID"] = project_id
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the Rockfish API."""
        url = f"{self.api_url}{endpoint}"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.content else {}
    
    async def call_endpoint(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool calls to appropriate API endpoints.

        This client handles operations that are not easily supported by the SDK:
        - Database operations (not in SDK)
        - Worker Set operations (not in SDK)
        - Dataset creation/update/schema (requires complex objects in SDK)
        - Model upload/delete (requires complex objects in SDK)
        - Project update (not in SDK)
        """

        # Database endpoints
        if tool_name == "list_databases":
            return await self._request("GET", "/database")

        elif tool_name == "create_database":
            return await self._request("POST", "/database", json=arguments)

        elif tool_name == "get_database":
            db_id = arguments["id"]
            return await self._request("GET", f"/database/{db_id}")

        elif tool_name == "update_database":
            db_id = arguments.pop("id")
            return await self._request("PUT", f"/database/{db_id}", json=arguments)

        elif tool_name == "delete_database":
            db_id = arguments["id"]
            return await self._request("DELETE", f"/database/{db_id}")

        # Worker Set endpoints
        elif tool_name == "list_worker_sets":
            return await self._request("GET", "/worker-set")

        elif tool_name == "create_worker_set":
            return await self._request("POST", "/worker-set", json=arguments)

        elif tool_name == "get_worker_set":
            ws_id = arguments["id"]
            return await self._request("GET", f"/worker-set/{ws_id}")

        elif tool_name == "delete_worker_set":
            ws_id = arguments["id"]
            return await self._request("DELETE", f"/worker-set/{ws_id}")

        elif tool_name == "get_worker_set_actions":
            ws_id = arguments["id"]
            return await self._request("GET", f"/worker-set/{ws_id}/actions")

        elif tool_name == "list_available_actions":
            worker_groups = await self._request("GET", "/worker-group")

            workers = []
            for group in worker_groups.get("groups", []):
                if "actions" in group:
                    current_actions = group.get("actions")
                    workers.extend(current_actions)

            return {
                "actions": workers
            }

        # Project update endpoint (SDK doesn't support update)
        elif tool_name == "update_project":
            project_id = arguments.pop("id")
            return await self._request("PATCH", f"/project/{project_id}", json=arguments)

        # Dataset endpoints (SDK requires complex objects for create/update)
        elif tool_name == "create_dataset":
            return await self._request("POST", "/dataset", json=arguments)

        elif tool_name == "update_dataset":
            dataset_id = arguments.pop("id")
            return await self._request("PATCH", f"/dataset/{dataset_id}", json=arguments)

        # Dataset schema endpoint (SDK doesn't have this)
        elif tool_name == "get_dataset_schema":
            dataset_id = arguments["id"]
            return await self._request("GET", f"/dataset/{dataset_id}/schema")

        # Model endpoints (SDK requires complex objects for upload)
        elif tool_name == "upload_model":
            return await self._request("POST", "/models", json=arguments)

        elif tool_name == "delete_model":
            model_id = arguments["id"]
            return await self._request("DELETE", f"/models/{model_id}")

        else:
            raise ValueError(f"Unknown tool: {tool_name}")