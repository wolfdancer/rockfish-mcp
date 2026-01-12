"""
Shared pytest fixtures for Rockfish MCP tests.

This module provides common fixtures used across multiple test files
to avoid duplication and ensure consistent test setup.
"""

import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv

import rockfish_mcp.server as server_module
from rockfish_mcp.client import RockfishClient
from rockfish_mcp.manta_client import MantaClient
from rockfish_mcp.sdk_client import RockfishSDKClient


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--env",
        action="store",
        default=".env",
        help="Path to the environment file to load (default: .env)",
    )


@pytest_asyncio.fixture(scope="session", autouse=True)
async def initialize_server_clients(request):
    """
    Initialize the server's global clients before running tests.

    This fixture runs once per test session and initializes the global
    rockfish_client, manta_client, and sdk_client in the server module
    so that handle_call_tool() can use them.
    """
    # Load environment variables from specified file
    # Use override=True to ensure the specified env file takes precedence
    env_file = request.config.getoption("--env")
    load_dotenv(env_file, override=True)

    api_key = os.getenv("ROCKFISH_API_KEY")
    api_url = os.getenv("ROCKFISH_API_URL", "https://api.rockfish.ai")
    manta_api_url = os.getenv("MANTA_API_URL")
    organization_id = os.getenv("ROCKFISH_ORGANIZATION_ID")
    project_id = os.getenv("ROCKFISH_PROJECT_ID")

    if not api_key:
        pytest.skip("ROCKFISH_API_KEY not set")

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

    # Only initialize Manta client if URL is configured
    if manta_api_url:
        server_module.manta_client = MantaClient(api_key=api_key, api_url=manta_api_url)

    yield

    # Cleanup
    if server_module.sdk_client:
        await server_module.sdk_client.close()


@pytest.fixture(scope="session")
def api_key():
    """Get API key from environment."""
    key = os.getenv("ROCKFISH_API_KEY")
    if not key:
        pytest.skip("ROCKFISH_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def organization_id():
    """Get organization ID from environment."""
    org_id = os.getenv("ROCKFISH_ORGANIZATION_ID")
    if not org_id:
        pytest.skip("ROCKFISH_ORGANIZATION_ID not set")
    return org_id


@pytest.fixture(scope="session")
def project_id():
    """Get project ID from environment."""
    proj_id = os.getenv("ROCKFISH_PROJECT_ID")
    if not proj_id:
        pytest.skip("ROCKFISH_PROJECT_ID not set")
    return proj_id


@pytest.fixture(scope="session")
def dataset_id():
    """Get dataset ID for testing from environment."""
    ds_id = os.getenv("INCIDENT_CREATION_TEST_DATASET")
    if not ds_id:
        pytest.skip("INCIDENT_CREATION_TEST_DATASET not set")
    return ds_id


@pytest.fixture(scope="session")
def manta_api_url():
    """Get Manta API URL from environment."""
    url = os.getenv("MANTA_API_URL")
    if not url:
        pytest.skip("MANTA_API_URL not set")
    return url
