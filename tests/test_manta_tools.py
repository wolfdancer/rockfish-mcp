"""Tests for Manta Analytics/Scenarios tool registration and routing."""

import pytest
from mcp import types

import rockfish_mcp.server as server_module
from rockfish_mcp.server import handle_call_tool, handle_list_tools


class DummyMantaClient:
    def __init__(self):
        self.calls = []

    async def call_endpoint(self, name, arguments):
        self.calls.append((name, arguments))
        return {"tool": name, "ok": True}


class DummyRockfishClient:
    def __init__(self):
        self.calls = []

    async def call_endpoint(self, name, arguments):
        self.calls.append((name, arguments))
        return {"tool": name, "ok": True}


@pytest.mark.asyncio
async def test_list_tools_registers_new_manta_tools():
    original_manta_client = server_module.manta_client
    try:
        server_module.manta_client = object()
        tools = await handle_list_tools()
    finally:
        server_module.manta_client = original_manta_client

    tool_names = {tool.name for tool in tools}
    assert "discover_schema" in tool_names
    assert "generate_test_suite" in tool_names
    assert "execute_query" in tool_names
    assert "execute_nl_query" in tool_names
    assert "inject_scenario" in tool_names

    assert "create_incident_dataset" not in tool_names
    assert "generate_incident_prompts" not in tool_names
    assert "list_incident_datasets" not in tool_names
    assert "get_incident_prompts" not in tool_names
    assert "execute_sql_query" in tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("discover_schema", {"dataset_id": "d1"}),
        ("generate_test_suite", {"dataset_id": "d1"}),
        ("execute_query", {"dataset_id": "d1", "query": {"aggregation": "Avg"}}),
        ("execute_nl_query", {"dataset_id": "d1", "question": "What is avg?"}),
        ("inject_scenario", {"dataset_id": "d1", "scenario": {"type": "spike"}}),
    ],
)
async def test_new_manta_tools_route_to_manta_client(tool_name, arguments):
    original_manta_client = server_module.manta_client
    original_rockfish_client = server_module.rockfish_client
    dummy_manta = DummyMantaClient()

    try:
        server_module.manta_client = dummy_manta
        server_module.rockfish_client = None
        result = await handle_call_tool(tool_name, arguments)
    finally:
        server_module.manta_client = original_manta_client
        server_module.rockfish_client = original_rockfish_client

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert dummy_manta.calls == [(tool_name, arguments)]
    assert tool_name in result[0].text


@pytest.mark.asyncio
async def test_execute_query_with_sql_string_routes_to_execute_sql_query():
    original_manta_client = server_module.manta_client
    original_rockfish_client = server_module.rockfish_client
    dummy_rockfish = DummyRockfishClient()

    try:
        server_module.manta_client = None
        server_module.rockfish_client = dummy_rockfish
        result = await handle_call_tool("execute_query", {"query": "SELECT 1"})
    finally:
        server_module.manta_client = original_manta_client
        server_module.rockfish_client = original_rockfish_client

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert dummy_rockfish.calls == [("execute_sql_query", {"query": "SELECT 1"})]
