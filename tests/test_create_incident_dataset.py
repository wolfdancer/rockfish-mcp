"""
Tests for the create_incident_dataset tool.

This test module parses incidents.yaml and creates one test for each
incident configuration defined in the file.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import pytest_asyncio
import yaml
from dotenv import load_dotenv
from mcp import types

import rockfish_mcp.server as server_module
from rockfish_mcp.client import RockfishClient
from rockfish_mcp.manta_client import MantaClient
from rockfish_mcp.sdk_client import RockfishSDKClient
from rockfish_mcp.server import handle_call_tool

# Load environment variables
load_dotenv()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def initialize_server_clients():
    """
    Initialize the server's global clients before running tests.

    This fixture runs once per test session and initializes the global
    rockfish_client, manta_client, and sdk_client in the server module
    so that handle_call_tool() can use them.
    """
    api_key = os.getenv("ROCKFISH_API_KEY")
    api_url = os.getenv("ROCKFISH_API_URL", "https://api.rockfish.ai")
    manta_api_url = os.getenv("MANTA_API_URL")
    organization_id = os.getenv("ROCKFISH_ORGANIZATION_ID")
    project_id = os.getenv("ROCKFISH_PROJECT_ID")

    if not api_key:
        pytest.skip("ROCKFISH_API_KEY not set")

    if not manta_api_url:
        pytest.skip("MANTA_API_URL not set")

    # Initialize global clients in the server module
    server_module.rockfish_client = RockfishClient(
        api_key=api_key,
        api_url=api_url,
        organization_id=organization_id,
        project_id=project_id,
    )

    server_module.sdk_client = RockfishSDKClient(
        API_KEY=api_key,
        API_URL=api_url,
        ORGANIZATION_ID=organization_id,
        PROJECT_ID=project_id,
    )

    server_module.manta_client = MantaClient(api_key=api_key, api_url=manta_api_url)

    yield

    # Cleanup
    if server_module.sdk_client:
        await server_module.sdk_client.close()


def load_incident_configs() -> List[Dict[str, Any]]:
    """
    Load incident configurations from incidents.yaml.

    Returns:
        List of incident configurations with type and configuration dict
    """
    incidents_file = Path(__file__).parent / "incidents.yaml"

    if not incidents_file.exists():
        pytest.skip(f"incidents.yaml not found at {incidents_file}")

    with open(incidents_file, "r") as f:
        incidents = yaml.safe_load(f)

    if not incidents:
        pytest.skip("No incidents found in incidents.yaml")

    return incidents


# Load incidents for parametrization
incident_configs = load_incident_configs()


class TestCreateIncidentDataset:
    """Test class for create_incident_dataset tool."""

    @pytest.fixture(scope="class")
    def organization_id(self):
        """Get organization ID from environment."""
        org_id = os.getenv("ROCKFISH_ORGANIZATION_ID")
        if not org_id:
            pytest.skip("ROCKFISH_ORGANIZATION_ID not set")
        return org_id

    @pytest.fixture(scope="class")
    def project_id(self):
        """Get project ID from environment."""
        proj_id = os.getenv("ROCKFISH_PROJECT_ID")
        if not proj_id:
            pytest.skip("ROCKFISH_PROJECT_ID not set")
        return proj_id

    @pytest.fixture(scope="class")
    def dataset_id(self):
        """Get dataset ID for testing from environment."""
        ds_id = os.getenv("INCIDENT_CREATION_TEST_DATASET")
        if not ds_id:
            pytest.skip("INCIDENT_CREATION_TEST_DATASET not set")
        return ds_id

    @pytest.mark.parametrize(
        "incident",
        incident_configs,
        ids=[inc["type"] for inc in incident_configs],
    )
    @pytest.mark.asyncio
    async def test_create_incident_dataset(
        self,
        organization_id: str,
        project_id: str,
        dataset_id: str,
        incident: Dict[str, Any],
    ):
        """
        Test creating an incident dataset with the given configuration.

        Args:
            organization_id: Organization ID for the request
            project_id: Project ID for the request
            dataset_id: Dataset ID for testing
            incident: Incident configuration from incidents.yaml
        """
        # Extract incident type and configuration
        incident_type = incident["type"]
        incident_config = incident["configuration"]

        # Prepare arguments for the create_incident_dataset tool
        create_arguments = {
            "dataset_id": dataset_id,
            "incident_type": incident_type,
            "incident_config": incident_config,
            "organization_id": organization_id,
            "project_id": project_id,
        }

        # Call the create_incident_dataset tool
        create_result = await handle_call_tool(
            "create_incident_dataset", create_arguments
        )

        # Assert that the call was successful
        assert create_result is not None, "Result should not be None"
        assert len(create_result) > 0, "Result should contain content"

        # Extract the text content from the MCP response
        first_content = create_result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text
        assert result_text, "Result text should not be empty"

        # Parse the response (server returns Python dict as string using str())
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try Python literal_eval (for dict strings with single quotes)
            result = ast.literal_eval(result_text)

        # Verify the response structure
        # The Manta API returns a response with dataset information
        created_dataset_id = None
        if isinstance(result, dict):
            # If single dataset response
            assert (
                "dataset_id" in result or "id" in result
            ), f"Response should contain dataset_id or id. Got: {result}"
            created_dataset_id = result.get("dataset_id") or result.get("id")
            assert created_dataset_id, "Created dataset ID should not be empty"

        elif isinstance(result, list):
            # If list response
            assert len(result) > 0, "Result list should not be empty"
            # Check first item has dataset info
            first_item = result[0]
            assert (
                "dataset_id" in first_item or "id" in first_item
            ), f"Response items should contain dataset_id or id. Got: {first_item}"
            created_dataset_id = first_item.get("dataset_id") or first_item.get("id")

        else:
            pytest.fail(f"Unexpected result type: {type(result)}. Result: {result}")

        # Log the created dataset for manual verification if needed
        print(f"\nCreated incident dataset for type '{incident_type}':")
        print(f"  Result: {result}")

        # Verify the dataset exists by retrieving it using get_dataset tool
        get_arguments = {"id": created_dataset_id}
        get_result = await handle_call_tool("get_dataset", get_arguments)

        assert get_result is not None, "get_dataset should return a result"
        assert len(get_result) > 0, "get_dataset should contain content"

        # Extract and parse the get_dataset response
        get_content = get_result[0]
        assert isinstance(
            get_content, types.TextContent
        ), "get_dataset result should be TextContent"
        get_result_text = get_content.text
        try:
            dataset_result = json.loads(get_result_text)
        except json.JSONDecodeError:
            dataset_result = ast.literal_eval(get_result_text)

        assert isinstance(dataset_result, dict), "get_dataset should return a dict"

        # Verify the retrieved dataset has the same ID
        retrieved_id = dataset_result.get("dataset_id") or dataset_result.get("id")
        assert (
            retrieved_id == created_dataset_id
        ), f"Retrieved dataset ID {retrieved_id} should match created ID {created_dataset_id}"

        print(f"  Verified dataset exists with ID: {retrieved_id}")

    @pytest.mark.asyncio
    async def test_invalid_incident_type(
        self,
        organization_id: str,
        project_id: str,
        dataset_id: str,
    ):
        """Test that invalid incident type returns an error message."""
        arguments = {
            "dataset_id": dataset_id,
            "incident_type": "invalid-incident-type",
            "incident_config": {"some": "config"},
            "organization_id": organization_id,
            "project_id": project_id,
        }

        # Call the tool - it should return an error message, not raise an exception
        result = await handle_call_tool("create_incident_dataset", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract the error message
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        error_text = first_content.text

        # Verify it's an error message
        assert (
            "Error calling create_incident_dataset" in error_text or "404" in error_text
        )

    @pytest.mark.asyncio
    async def test_missing_required_config_field(
        self,
        organization_id: str,
        project_id: str,
        dataset_id: str,
    ):
        """Test that missing required configuration fields returns an error message."""
        arguments = {
            "dataset_id": dataset_id,
            "incident_type": "instantaneous-spike-data",
            "incident_config": {
                # Missing required fields like impacted_measurement, absolute_magnitude, etc.
                "timestamp_column": "timestamp",
            },
            "organization_id": organization_id,
            "project_id": project_id,
        }

        # Call the tool - it should return an error message, not raise an exception
        result = await handle_call_tool("create_incident_dataset", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract the error message
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        error_text = first_content.text

        # Verify it's an error message (500 error due to validation failure)
        assert (
            "Error calling create_incident_dataset" in error_text or "500" in error_text
        )
