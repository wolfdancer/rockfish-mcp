# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup

**IMPORTANT**: Always use a virtual environment. Python 3.11 is recommended (SDK requires 3.12 or below).

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install in development mode with dev dependencies (includes black and isort)
pip install -e ".[dev]"

# OR install from requirements.txt (exact locked versions)
pip install -r requirements.txt
```

**Important**: The Rockfish SDK is hosted at `https://packages.rockfish.ai`. The [requirements.txt](requirements.txt) uses `--find-links https://packages.rockfish.ai` to resolve it.

### Environment Setup
Required in a `.env` file at the project root:
- `ROCKFISH_API_KEY`: API key (required)
- `ROCKFISH_API_URL`: Defaults to `https://api.rockfish.ai`
- `ROCKFISH_ORGANIZATION_ID`: Falls back to the account default if unset
- `ROCKFISH_PROJECT_ID`: Falls back to the account default if unset
- `MANTA_API_URL`: Enables Manta tools when set (e.g. `https://manta.rockfish.ai`)

```bash
cp .env.example .env
```

### Running the Server
```bash
python -m rockfish_mcp.server
# or
rockfish-mcp
```

### Code Formatting
```bash
isort src/rockfish_mcp/ && black src/rockfish_mcp/

# Check only
isort --check-only src/rockfish_mcp/ && black --check src/rockfish_mcp/
```

### Running Tests

The test suite has two tiers:

**Unit tests** — no credentials required, run locally:
```bash
pytest tests/test_manta_client.py tests/test_manta_tools.py
```

**Integration tests** — require a `.env` file with real credentials:
```bash
# Run all tests against the default .env
pytest tests/

# Target a specific environment file
pytest tests/ --env=.env.staging.local

# Run a specific test
pytest tests/test_list_resources.py::TestListResources::test_list_databases
```

Test files:
- [tests/test_manta_client.py](tests/test_manta_client.py): Unit tests for Manta auth headers and endpoint routing (no API needed)
- [tests/test_manta_tools.py](tests/test_manta_tools.py): Unit tests for tool registration, server-side routing, and SQL query backward compatibility (no API needed)
- [tests/test_list_resources.py](tests/test_list_resources.py): Integration tests for all `list_*` tools
- [tests/conftest.py](tests/conftest.py): Session-scoped fixtures; initializes global server clients from env vars; provides `--env` CLI option

### Testing with MCP Inspector
```bash
npx @modelcontextprotocol/inspector /path/to/.venv/bin/python -m rockfish_mcp.server
```

## Architecture Overview

An MCP server that bridges AI assistants to three backends: the Rockfish REST API, the Manta analytics/scenario service, and the Rockfish Python SDK.

```
src/rockfish_mcp/
├── server.py       # Tool definitions, registration, and routing
├── client.py       # HTTP client for Rockfish REST API
├── manta_client.py # HTTP client for Manta service
└── sdk_client.py   # Wrapper for Rockfish Python SDK
```

### Tool Routing in server.py

`handle_call_tool()` dispatches tool calls via three sequential checks:

1. **SDK tools** (`sdk_tools` list) → `sdk_client.call_endpoint()`
2. **Manta tools** (`manta_tools` list) → `manta_client.call_endpoint()`
   - Before dispatch, `organization_id` and `project_id` are injected from `ROCKFISH_ORGANIZATION_ID` / `ROCKFISH_PROJECT_ID` env vars if not provided in the arguments
   - `execute_query` with a string `query` argument is rewritten to `execute_sql_query` for backward compatibility
3. **Everything else** → `rockfish_client.call_endpoint()`

Manta tools are only registered (in `handle_list_tools()`) and only routable when `manta_client` is initialized, which requires `MANTA_API_URL` to be set at startup.

### Client Patterns

All three clients expose a single `call_endpoint(tool_name, arguments)` method with `if/elif` dispatch internally. HTTP clients (`RockfishClient`, `MantaClient`) use `httpx.AsyncClient`. The SDK client uses `rf.Connection.remote()` from the native Rockfish Python SDK.

**MantaClient** sends `X-Organization-Id` and `X-Project-Id` as request headers, populated from the `organization_id` / `project_id` keys in `arguments`.

**RockfishSDKClient** maintains an in-memory UUID-keyed cache of training configurations between calls, enabling a multi-step workflow: `obtain_train_config` → `start_training_workflow` → `get_trained_model_id` → `start_generation_workflow` → `obtain_synthetic_dataset_id`.

### Tool Categories

| Category | Always available | Tools |
|---|---|---|
| Rockfish API | Yes | Databases, Worker Sets, Worker Groups, Workflows, Models, Projects, Datasets, Organizations (22+ tools) |
| SDK | Yes | `obtain_train_config`, `start_training_workflow`, `get_workflow_logs`, `get_trained_model_id`, `start_generation_workflow`, `obtain_synthetic_dataset_id`, `plot_distribution`, `get_marginal_distribution_score`, `update_train_config` |
| Manta | Only if `MANTA_API_URL` set | `discover_schema`, `generate_test_suite`, `execute_query`, `execute_nl_query`, `inject_scenario` |

## API Reference

- **Rockfish API**: https://docs.rockfish.ai/openapi.yaml
- **Manta Service**: https://manta.rockfish.ai/openapi.json
