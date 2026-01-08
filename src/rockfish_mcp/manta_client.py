"""
Manta service client for Rockfish MCP.

Handles API calls to the Manta service for dataset pattern injection
and test case generation.
"""

import os
from typing import Any

import httpx


class MantaClient:
    """Client for interacting with the Rockfish Manta API."""

    def __init__(self, api_key: str, api_url: str = "https://manta.rockfish.ai"):
        """
        Initialize the Manta client.

        Args:
            api_key: Rockfish API key for authentication
            api_url: Base URL for the Manta API (default: https://manta.rockfish.ai)
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.headers = {
            "X-API-Key": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def call_endpoint(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Call a Manta API endpoint based on the tool name.

        Args:
            tool_name: Name of the MCP tool being called
            arguments: Tool arguments containing request parameters

        Returns:
            API response data

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        async with httpx.AsyncClient() as client:
            # Extract common headers required by Manta API
            extra_headers = {}
            if "organization_id" in arguments:
                extra_headers["X-Organization-ID"] = arguments["organization_id"]
            if "project_id" in arguments:
                extra_headers["X-Project-ID"] = arguments["project_id"]

            headers = {**self.headers, **extra_headers}

            # Route to appropriate endpoint based on tool name
            # V2 Incident Injection Tools
            if tool_name == "create_incident_dataset":
                incident_type = arguments["incident_type"]
                response = await client.post(
                    f"{self.api_url}/{incident_type}",
                    headers=headers,
                    json={
                        "dataset_id": arguments["dataset_id"],
                        "incident_config": arguments["incident_config"],
                    },
                )

            elif tool_name == "generate_incident_prompts":
                response = await client.post(
                    f"{self.api_url}/prompts",
                    headers=headers,
                    json={"dataset_id": arguments["dataset_id"]},
                )

            elif tool_name == "list_incident_datasets":
                response = await client.post(
                    f"{self.api_url}/incident-dataset-ids",
                    headers=headers,
                    json={"dataset_id": arguments["dataset_id"]},
                )

            elif tool_name == "get_incident_prompts":
                dataset_id = arguments["dataset_id"]
                response = await client.get(
                    f"{self.api_url}/prompts",
                    headers=headers,
                    params={"dataset_id": dataset_id},
                )

            else:
                raise ValueError(f"Unknown Manta tool: {tool_name}")

            response.raise_for_status()
            return response.json()
