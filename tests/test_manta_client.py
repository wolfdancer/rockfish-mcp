"""Unit tests for MantaClient endpoint routing and headers."""

import pytest

import rockfish_mcp.manta_client as manta_client_module
from rockfish_mcp.manta_client import MantaClient


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyAsyncClient:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return DummyResponse({"ok": True, "url": url})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,arguments,expected_path,expected_payload",
    [
        (
            "discover_schema",
            {"dataset_id": "d1", "organization_id": "o1", "project_id": "p1"},
            "/analytics/discover-schema",
            {"dataset_id": "d1"},
        ),
        (
            "generate_test_suite",
            {
                "csv_content": "a,b\n1,2",
                "schema": {"timestamp_column": "ts"},
                "organization_id": "o1",
                "project_id": "p1",
            },
            "/analytics/generate-test-suite",
            {"csv_content": "a,b\n1,2", "schema": {"timestamp_column": "ts"}},
        ),
        (
            "execute_query",
            {
                "dataset_id": "d1",
                "query": {"aggregation": "Avg"},
                "include_questions": True,
                "organization_id": "o1",
                "project_id": "p1",
            },
            "/analytics/execute-query",
            {
                "dataset_id": "d1",
                "query": {"aggregation": "Avg"},
                "include_questions": True,
            },
        ),
        (
            "execute_nl_query",
            {
                "dataset_id": "d1",
                "question": "What is avg?",
                "timestamp_column": "ts",
                "organization_id": "o1",
                "project_id": "p1",
            },
            "/analytics/execute-nl-query",
            {
                "dataset_id": "d1",
                "question": "What is avg?",
                "timestamp_column": "ts",
            },
        ),
        (
            "inject_scenario",
            {
                "dataset_id": "d1",
                "scenario": {"type": "spike"},
                "generate_tests": True,
                "max_cases": 20,
                "organization_id": "o1",
                "project_id": "p1",
            },
            "/scenarios/inject",
            {
                "dataset_id": "d1",
                "scenario": {"type": "spike"},
                "generate_tests": True,
                "max_cases": 20,
            },
        ),
    ],
)
async def test_call_endpoint_routes_to_expected_path_and_headers(
    monkeypatch, tool_name, arguments, expected_path, expected_payload
):
    dummy_client = DummyAsyncClient()
    monkeypatch.setattr(manta_client_module.httpx, "AsyncClient", lambda: dummy_client)

    client = MantaClient(api_key="test-key", api_url="https://manta.example")
    result = await client.call_endpoint(tool_name, arguments)

    assert result["ok"] is True
    assert len(dummy_client.calls) == 1
    call = dummy_client.calls[0]
    assert call["url"] == f"https://manta.example{expected_path}"
    assert call["json"] == expected_payload

    headers = call["headers"]
    assert headers["X-Api-Key"] == "Bearer test-key"
    assert headers["X-Organization-Id"] == "o1"
    assert headers["X-Project-Id"] == "p1"
    assert "X-API-Key" not in headers
    assert "X-Organization-ID" not in headers
    assert "X-Project-ID" not in headers


@pytest.mark.asyncio
async def test_call_endpoint_raises_for_unknown_tool():
    client = MantaClient(api_key="test-key", api_url="https://manta.example")
    with pytest.raises(ValueError, match="Unknown Manta tool"):
        await client.call_endpoint("unknown_tool", {})
