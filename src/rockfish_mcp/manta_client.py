"""Manta service client for Rockfish MCP.

Handles Analytics and Scenarios API calls for schema discovery,
test-suite generation, querying, and scenario injection.
"""

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
            "X-Api-Key": f"Bearer {api_key}",
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
            extra_headers = {}
            if "organization_id" in arguments:
                extra_headers["X-Organization-Id"] = arguments["organization_id"]
            if "project_id" in arguments:
                extra_headers["X-Project-Id"] = arguments["project_id"]

            headers = {**self.headers, **extra_headers}

            if tool_name == "discover_schema":
                response = await client.post(
                    f"{self.api_url}/analytics/discover-schema",
                    headers=headers,
                    json=self._build_payload(arguments, ["dataset_id", "csv_content"]),
                )

            elif tool_name == "generate_test_suite":
                response = await client.post(
                    f"{self.api_url}/analytics/generate-test-suite",
                    headers=headers,
                    json=self._build_payload(
                        arguments, ["dataset_id", "csv_content", "schema"]
                    ),
                )

            elif tool_name == "execute_query":
                response = await client.post(
                    f"{self.api_url}/analytics/execute-query",
                    headers=headers,
                    json=self._build_payload(
                        arguments,
                        ["dataset_id", "query", "timestamp_column", "include_questions"],
                    ),
                )

            elif tool_name == "execute_nl_query":
                response = await client.post(
                    f"{self.api_url}/analytics/execute-nl-query",
                    headers=headers,
                    json=self._build_payload(
                        arguments,
                        ["dataset_id", "question", "schema", "timestamp_column"],
                    ),
                )

            elif tool_name == "inject_scenario":
                response = await client.post(
                    f"{self.api_url}/scenarios/inject",
                    headers=headers,
                    json=self._build_payload(
                        arguments,
                        [
                            "dataset_id",
                            "csv_content",
                            "scenario",
                            "generate_tests",
                            "include_negative",
                            "max_cases",
                            "variations_per_question",
                        ],
                    ),
                )

            else:
                raise ValueError(f"Unknown Manta tool: {tool_name}")

            response.raise_for_status()
            return response.json()

    @staticmethod
    def _build_payload(arguments: dict[str, Any], allowed_keys: list[str]) -> dict[str, Any]:
        """Return a payload with only endpoint-supported keys."""
        return {key: arguments[key] for key in allowed_keys if key in arguments}
