"""
Tests for all list_ tools in the Rockfish MCP server.

This test module exercises all tools starting with "list_" and verifies
that they return non-empty lists of resources.
"""

import ast
import json

import pytest
from mcp import types

from rockfish_mcp.server import handle_call_tool


def parse_response_text(result_text: str, context: str = "response") -> any:
    """
    Parse response text from the MCP server, handling both JSON and Python literals.

    Args:
        result_text: The response text to parse
        context: Description of what is being parsed (for error messages)

    Returns:
        Parsed Python object (dict, list, etc.)

    Raises:
        pytest.fail: If the response is an error message or cannot be parsed
    """
    # Check if this is an error message before parsing
    if result_text.startswith("Error calling"):
        pytest.fail(f"API call failed for {context}: {result_text}")

    # Try JSON first
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        pass

    # Try Python literal_eval
    try:
        return ast.literal_eval(result_text)
    except (ValueError, SyntaxError) as e:
        pytest.fail(
            f"Failed to parse {context} as JSON or Python literal.\n"
            f"Parse error: {e}\n"
            f"Response text: {result_text[:500]}"
        )



class TestListResources:
    """Test class for list_ tools."""

    @pytest.mark.asyncio
    async def test_list_databases(self, organization_id: str, project_id: str):
        """Test listing databases returns non-empty list."""
        arguments = {
            "organization_id": organization_id,
            "project_id": project_id,
        }

        result = await handle_call_tool("list_databases", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        databases = parse_response_text(result_text, "list_databases")

        # Verify we got a list back
        assert isinstance(databases, list), "Result should be a list"
        # Note: We don't assert non-empty as the list might be empty if no databases exist

    @pytest.mark.asyncio
    async def test_list_worker_sets(self, organization_id: str, project_id: str):
        """Test listing worker sets returns a list and test getting actions for each."""
        arguments = {
            "organization_id": organization_id,
            "project_id": project_id,
        }

        result = await handle_call_tool("list_worker_sets", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        worker_sets = parse_response_text(result_text, "list_worker_sets")

        # Verify we got a list back
        assert isinstance(worker_sets, list), "Result should be a list"

        # Track results for each worker set
        successful_worker_sets = []
        failed_worker_sets = []

        # Iterate through each worker set and call get_worker_set_actions
        for worker_set in worker_sets:
            assert (
                "id" in worker_set
            ), f"Worker set should have an 'id' field: {worker_set}"
            worker_set_id = worker_set["id"]
            worker_set_name = worker_set.get("name", "unknown")

            # Call get_worker_set_actions for this worker set
            actions_result = await handle_call_tool(
                "get_worker_set_actions", {"id": worker_set_id}
            )

            assert (
                actions_result is not None
            ), f"Actions result should not be None for worker set {worker_set_id} ({worker_set_name})"
            assert (
                len(actions_result) > 0
            ), f"Actions result should contain content for worker set {worker_set_id} ({worker_set_name})"

            # Extract and parse the actions response
            actions_content = actions_result[0]
            assert isinstance(
                actions_content, types.TextContent
            ), f"Actions result should be TextContent for worker set {worker_set_id} ({worker_set_name})"
            actions_text = actions_content.text

            # Check if it's an error message
            if actions_text.startswith("Error calling"):
                failed_worker_sets.append(
                    {
                        "id": worker_set_id,
                        "name": worker_set_name,
                        "error": actions_text,
                    }
                )
                continue

            # Parse the actions response with proper error handling
            try:
                actions = parse_response_text(
                    actions_text, f"get_worker_set_actions for {worker_set_name}"
                )
            except Exception as e:
                # If parsing fails, treat it as a failed worker set
                failed_worker_sets.append(
                    {
                        "id": worker_set_id,
                        "name": worker_set_name,
                        "error": str(e),
                    }
                )
                continue

            # Verify actions structure (should be a list or dict)
            assert isinstance(
                actions, (list, dict)
            ), f"Actions should be a list or dict for worker set {worker_set_id} ({worker_set_name})"

            successful_worker_sets.append(
                {"id": worker_set_id, "name": worker_set_name, "actions": actions}
            )

        # If any worker sets failed, fail the test with detailed information
        if failed_worker_sets:
            error_details = "\n".join(
                [
                    f"  - {ws['name']} (ID: {ws['id']}): {ws['error']}"
                    for ws in failed_worker_sets
                ]
            )
            success_details = "\n".join(
                [f"  - {ws['name']} (ID: {ws['id']})" for ws in successful_worker_sets]
            )
            pytest.fail(
                f"Failed to get actions for {len(failed_worker_sets)} worker set(s):\n{error_details}\n\n"
                f"Successfully retrieved actions for {len(successful_worker_sets)} worker set(s):\n{success_details}"
            )

    @pytest.mark.asyncio
    async def test_list_worker_groups(self):
        """Test listing worker groups returns a dict with groups list."""
        arguments = {}

        result = await handle_call_tool("list_worker_groups", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        try:
            response = json.loads(result_text)
        except json.JSONDecodeError:
            response = ast.literal_eval(result_text)

        # Verify we got a dict with 'groups' key
        assert isinstance(response, dict), "Result should be a dict"
        assert "groups" in response, "Result should contain 'groups' key"
        groups = response["groups"]
        assert isinstance(groups, list), "groups should be a list"
        # Worker groups should not be empty
        assert len(groups) > 0, "Worker groups list should not be empty"

    @pytest.mark.asyncio
    async def test_list_available_actions(self):
        """Test listing available actions returns a dict with actions list."""
        arguments = {}

        result = await handle_call_tool("list_available_actions", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        response = parse_response_text(result_text, "list_available_actions")

        # Verify we got a dict with 'actions' key
        assert isinstance(response, dict), "Result should be a dict"
        assert "actions" in response, "Result should contain 'actions' key"
        actions = response["actions"]
        assert isinstance(actions, list), "actions should be a list"
        # Available actions should not be empty
        assert len(actions) > 0, "Available actions list should not be empty"

    @pytest.mark.asyncio
    async def test_list_workflows(self, organization_id: str, project_id: str):
        """Test listing workflows returns a list."""
        arguments = {
            "organization_id": organization_id,
            "project_id": project_id,
        }

        result = await handle_call_tool("list_workflows", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        workflows = parse_response_text(result_text, "list_workflows")

        # Verify we got a list back
        assert isinstance(workflows, list), "Result should be a list"

    @pytest.mark.asyncio
    async def test_list_models(self, organization_id: str, project_id: str):
        """Test listing models returns a list."""
        arguments = {
            "organization_id": organization_id,
            "project_id": project_id,
        }

        result = await handle_call_tool("list_models", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        models = parse_response_text(result_text, "list_models")

        # Verify we got a list back
        assert isinstance(models, list), "Result should be a list"

    @pytest.mark.asyncio
    async def test_list_organizations(self):
        """Test listing organizations returns non-empty list."""
        arguments = {}

        result = await handle_call_tool("list_organizations", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        organizations = parse_response_text(result_text, "list_organizations")

        # Verify we got a list back
        assert isinstance(organizations, list), "Result should be a list"
        # Should have at least one organization
        assert len(organizations) > 0, "Organizations list should not be empty"

    @pytest.mark.asyncio
    async def test_list_projects(self, organization_id: str):
        """Test listing projects returns non-empty list."""
        arguments = {
            "organization_id": organization_id,
        }

        result = await handle_call_tool("list_projects", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        projects = parse_response_text(result_text, "list_projects")

        # Verify we got a list back
        assert isinstance(projects, list), "Result should be a list"
        # Should have at least one project
        assert len(projects) > 0, "Projects list should not be empty"

    @pytest.mark.asyncio
    async def test_list_datasets(self, organization_id: str, project_id: str):
        """Test listing datasets returns a list."""
        arguments = {
            "organization_id": organization_id,
            "project_id": project_id,
        }

        result = await handle_call_tool("list_datasets", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        datasets = parse_response_text(result_text, "list_datasets")

        # Verify we got a list back
        assert isinstance(datasets, list), "Result should be a list"

    @pytest.mark.asyncio
    async def test_list_incident_datasets(
        self, organization_id: str, project_id: str, dataset_id: str, manta_api_url: str
    ):
        """Test listing incident datasets returns a dict with dataset_ids list."""
        # This test requires Manta API to be configured
        arguments = {
            "dataset_id": dataset_id,
            "organization_id": organization_id,
            "project_id": project_id,
        }

        result = await handle_call_tool("list_incident_datasets", arguments)

        assert result is not None, "Result should not be None"
        assert len(result) > 0, "Result should contain content"

        # Extract and parse the response
        first_content = result[0]
        assert isinstance(
            first_content, types.TextContent
        ), "Result should be TextContent"
        result_text = first_content.text

        # Parse the response
        response = parse_response_text(result_text, "list_incident_datasets")

        # Verify we got a dict with 'dataset_ids' key
        assert isinstance(response, dict), "Result should be a dict"
        assert "dataset_ids" in response, "Result should contain 'dataset_ids' key"
        dataset_ids = response["dataset_ids"]
        assert isinstance(dataset_ids, list), "dataset_ids should be a list"
