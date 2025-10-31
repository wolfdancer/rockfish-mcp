"""
Manta service client for Rockfish MCP.

Handles API calls to the Manta service for dataset pattern injection
and test case generation.
"""

import os
import httpx
from typing import Any


class MantaClient:
    """Client for interacting with the Rockfish Manta API."""

    def __init__(self, api_key: str, base_url: str = "https://manta.sunset-beach.rockfish.ai"):
        """
        Initialize the Manta client.

        Args:
            api_key: Rockfish API key for authentication
            base_url: Base URL for the Manta API (default: https://manta.sunset-beach.rockfish.ai)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def call_endpoint(self, tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
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
            if tool_name == "manta_get_prompts":
                dataset_id = arguments["dataset_id"]
                response = await client.get(
                    f"{self.base_url}/prompts",
                    headers=headers,
                    params={"dataset_id": dataset_id}
                )

            elif tool_name == "manta_create_prompts":
                response = await client.post(
                    f"{self.base_url}/prompts",
                    headers=headers,
                    json={"dataset_id": arguments["dataset_id"]}
                )

            elif tool_name == "manta_append_prompts":
                response = await client.patch(
                    f"{self.base_url}/prompts",
                    headers=headers,
                    json={"dataset_id": arguments["dataset_id"]}
                )

            elif tool_name == "manta_evaluate_test_case":
                response = await client.post(
                    f"{self.base_url}/evaluate-test-case",
                    headers=headers,
                    json={
                        "prompt": arguments["prompt"],
                        "actual_result": arguments["actual_result"],
                        "expected_result": arguments["expected_result"]
                    }
                )

            elif tool_name == "manta_create_instantaneous_spike":
                response = await client.post(
                    f"{self.base_url}/instantaneous-spike-data",
                    headers=headers,
                    json={
                        "dataset_id": arguments["dataset_id"],
                        "incident_config": arguments["incident_config"]
                    }
                )

            elif tool_name == "manta_create_sustained_magnitude_change":
                response = await client.post(
                    f"{self.base_url}/sustained-magnitude-change-data",
                    headers=headers,
                    json={
                        "dataset_id": arguments["dataset_id"],
                        "incident_config": arguments["incident_config"]
                    }
                )

            elif tool_name == "manta_create_data_outage":
                response = await client.post(
                    f"{self.base_url}/data-outage-data",
                    headers=headers,
                    json={
                        "dataset_id": arguments["dataset_id"],
                        "incident_config": arguments["incident_config"]
                    }
                )

            elif tool_name == "manta_create_value_ramp":
                response = await client.post(
                    f"{self.base_url}/value-ramp-data",
                    headers=headers,
                    json={
                        "dataset_id": arguments["dataset_id"],
                        "incident_config": arguments["incident_config"]
                    }
                )

            elif tool_name == "manta_get_incident_dataset_ids":
                response = await client.post(
                    f"{self.base_url}/incident-dataset-ids",
                    headers=headers,
                    json={"dataset_id": arguments["dataset_id"]}
                )

            elif tool_name == "manta_process_llm_questions":
                response = await client.post(
                    f"{self.base_url}/customer-llm",
                    headers=headers,
                    json={
                        "dataset_id": arguments["dataset_id"],
                        "questions": arguments["questions"]
                    }
                )

            else:
                raise ValueError(f"Unknown Manta tool: {tool_name}")

            response.raise_for_status()
            return response.json()
