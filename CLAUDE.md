# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup

**IMPORTANT**: Always use a virtual environment when running Python scripts in this project. Prefer using the existing virtual environment at the project root if it exists (commonly named `.venv/` or `venv/`).

```bash
# Create and activate virtual environment if it doesn't exist
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows

# Install in development mode with dev dependencies (includes black and isort)
pip install -e ".[dev]"

# OR install from requirements.txt (exact locked versions for development)
pip install -r requirements.txt
```

**Important**: The Rockfish SDK package is hosted on a custom package repository at `https://packages.rockfish.ai`. The [requirements.txt](requirements.txt) file uses `--find-links https://packages.rockfish.ai` to enable pip to discover and install packages from this repository.

### Running the Server
```bash
# Run the MCP server directly
python -m rockfish_mcp.server

# Or use the console script
rockfish-mcp
```

### Environment Setup
The application requires these environment variables (configured in `.env` file):
- `ROCKFISH_API_KEY`: Your Rockfish API key (required)
- `ROCKFISH_API_URL`: API URL for Rockfish API (optional, defaults to https://api.rockfish.ai)
- `ROCKFISH_ORGANIZATION_ID`: Organization ID (optional - uses default if not set)
- `ROCKFISH_PROJECT_ID`: Project ID (optional - uses default if not set)
- `MANTA_API_URL`: API URL for Manta service (optional - Manta tools only appear if this is set)
- `INCIDENT_CREATION_TEST_DATASET`: Dataset ID for incident injection testing (optional - required only for running tests)

Create a `.env` file with these variables for local development:
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Code Formatting
This project uses black and isort for code formatting:
```bash
# Format code before committing
isort src/rockfish_mcp/
black src/rockfish_mcp/

# Check formatting without modifying files
isort --check-only src/rockfish_mcp/
black --check src/rockfish_mcp/
```

### Testing with MCP Inspector
Use the MCP Inspector to test the server before connecting to Claude Desktop:
```bash
# Start the inspector (replace with your actual Python path)
npx @modelcontextprotocol/inspector /path/to/.venv/bin/python -m rockfish_mcp.server
```
The Inspector provides an interactive web interface to test all available tools.

### Running Tests
The project includes automated tests for both list tools and Manta incident injection tools:
```bash
# Run all tests (uses .env by default)
pytest tests/

# Run tests with a specific environment file (via custom --env option)
pytest tests/ --env=.env.staging.local
pytest tests/ --env=.env.prod.local

# Run specific test file
pytest tests/test_list_resources.py
pytest tests/test_create_incident_dataset.py
pytest tests/test_workerset_actions.py

# Run specific test class or method
pytest tests/test_list_resources.py::TestListResources::test_list_databases
pytest tests/test_create_incident_dataset.py::TestIncidentCreation::test_instantaneous_spike

# Run tests with verbose output
pytest -v tests/

# Run tests with print output visible
pytest -s tests/

# Generate HTML test report
pytest tests/ --html=report.html --self-contained-html
```

**Test Requirements:**
- `ROCKFISH_API_KEY`: Required for authentication (all tests)
- `ROCKFISH_API_URL`: API endpoint (optional, defaults to https://api.rockfish.ai)
- `ROCKFISH_ORGANIZATION_ID`: Organization ID (required for most tests)
- `ROCKFISH_PROJECT_ID`: Project ID (required for most tests)
- `MANTA_API_URL`: Manta service endpoint (required for Manta tests only)
- `INCIDENT_CREATION_TEST_DATASET`: Dataset ID to use for incident injection testing (required for create_incident_dataset tests only)

**Test Structure:**
- [tests/conftest.py](tests/conftest.py): Shared pytest fixtures for client initialization and environment setup (includes custom `--env` option for loading different .env files)
- [tests/test_list_resources.py](tests/test_list_resources.py): Tests all list_ tools (list_databases, list_worker_sets, list_workflows, list_models, list_projects, list_datasets, list_organizations, list_worker_groups, list_available_actions, list_incident_datasets)
- [tests/test_create_incident_dataset.py](tests/test_create_incident_dataset.py): Tests the create_incident_dataset tool with configurations from [tests/incidents.yaml](tests/incidents.yaml)
- [tests/test_workerset_actions.py](tests/test_workerset_actions.py): Tests worker set action listing functionality
- [tests/incidents.yaml](tests/incidents.yaml): Defines test cases for all four incident types (instantaneous-spike, sustained-magnitude-change, data-outage, value-ramp)

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides AI assistants access to the Rockfish machine learning platform API, the Manta incident injection service, and the Rockfish SDK for synthetic data generation. The architecture consists of four main components in a simple, focused structure.

### Project Structure
```
src/rockfish_mcp/
├── __init__.py
├── server.py       # MCP server with tool definitions and routing
├── client.py       # HTTP client for Rockfish API calls
├── manta_client.py # HTTP client for Manta service calls
└── sdk_client.py   # SDK client for Rockfish python SDK calls
```

### Core Components

**Server ([server.py](src/rockfish_mcp/server.py))**: The main MCP server that:
- Defines tools across multiple resource categories
  - Rockfish API: Databases, Worker Sets, Worker Groups, Workflows, Models, Projects, Datasets, Organizations (22+ tools, always available)
  - Manta Service: Incident Injection and Prompt Management (4 tools, conditional)
  - SDK Tools: Synthetic Data Generation workflow tools (9 tools, always available)
- Conditionally loads Manta tools only when `MANTA_API_URL` environment variable is set
- Handles tool registration via `@server.list_tools()` decorator
- Routes tool calls through `@server.call_tool()` decorator:
  - SDK tools (in `sdk_tools` list) route to `sdk_client`
  - Manta tools (in `manta_v2_tools` list) route to `manta_client`
  - All other tools route to `rockfish_client`
- Manages server initialization and stdio communication with MCP protocol
- Uses global `rockfish_client` (always), `sdk_client` (always), and `manta_client` (conditional) instances initialized in `main()`
- Requires `ROCKFISH_API_KEY` environment variable to function

**Client ([client.py](src/rockfish_mcp/client.py))**: HTTP client wrapper for Rockfish API that:
- Handles Bearer token authentication for all API requests
- Provides async HTTP requests to Rockfish API endpoints via httpx
- Maps MCP tool names to specific HTTP endpoints and methods in `call_endpoint()`
- Uses different HTTP methods (GET, POST, PUT, PATCH, DELETE) based on operation
- Centralizes error handling with `raise_for_status()` and returns formatted responses

**Manta Client ([manta_client.py](src/rockfish_mcp/manta_client.py))**: HTTP client wrapper for Manta service that:
- Handles Bearer token authentication (uses same `ROCKFISH_API_KEY`)
- Provides async HTTP requests to Manta service endpoints via httpx
- Manages required Manta headers (`X-Organization-ID`, `X-Project-ID`)
- Maps Manta tool names to specific endpoints for:
  - Incident injection v2 (unified tool with type parameter)
  - Prompt management (generate and retrieve prompts for incident datasets)
  - Dataset listing (list all incident datasets from a source)
- Centralizes error handling and returns formatted responses

**SDK Client ([sdk_client.py](src/rockfish_mcp/sdk_client.py))**: Python SDK wrapper for Rockfish workflows that:
- Uses native Rockfish Python SDK (`rockfish` package) instead of HTTP calls
- Provides direct access to Rockfish SDK connection via `rf.Connection.remote()`
- Manages synthetic data generation workflow from end-to-end:
  - Training configuration generation with automatic column type detection
  - Rockfish TabGAN model training workflow execution
  - Model extraction and synthetic data generation
  - Distribution plotting and quality metrics
- Maintains in-memory cache for training configurations (using UUIDs)
- Implements streaming workflow logs with configurable log levels and timeouts
- Returns structured responses with `success` flags and detailed error messages
- Uses PyArrow for efficient dataset manipulation

### Tool Categories and API Mapping

#### Rockfish API Endpoints
The server exposes CRUD operations mapping to these endpoints:
- **Databases**: `/database` endpoints (GET, POST, PUT, DELETE)
- **Worker Sets**: `/worker-set` endpoints (GET, POST, DELETE - no update)
- **Worker Groups**: `/worker-group` endpoints (GET list only)
- **Workflows**: `/workflow` endpoints (GET, POST, PUT)
- **Models**: `/models` endpoints (GET, POST, DELETE - note different path)
- **Projects**: `/project` endpoints (GET, POST, PATCH)
- **Datasets**: `/dataset` endpoints (GET, POST, PATCH, DELETE)
- **Organizations**: `/organization` endpoints (GET current org)

#### Manta Service Endpoints (Optional, v2)
The Manta service provides incident injection capabilities for time-series datasets. These tools are only available when `MANTA_API_URL` is configured:
- **Incident Injection v2** (unified interface):
  - `create_incident_dataset`: Creates incident datasets via `POST /{incident_type}` where incident_type is one of:
    - `instantaneous-spike-data`: Inject sudden spikes
    - `sustained-magnitude-change-data`: Apply sustained changes
    - `data-outage-data`: Create data gaps
    - `value-ramp-data`: Apply gradual changes
- **Prompt Management**:
  - `generate_incident_prompts`: Generate analysis prompts via `POST /prompts`
  - `get_incident_prompts`: Retrieve prompts via `GET /prompts`
- **Dataset Management**:
  - `list_incident_datasets`: List all incident datasets via `POST /incident-dataset-ids`

#### SDK Tools (Rockfish Python SDK)
The SDK client provides end-to-end synthetic data generation workflow tools using the native Rockfish Python SDK:
- **Configuration Management**:
  - `obtain_train_config`: Generate training config with automatic column type detection (categorical/continuous/high-cardinality)
  - `update_train_config` [experimental]: Modify hyperparameters (epochs, batch_size, learning rates) or field classifications
- **Workflow Execution**:
  - `start_training_workflow`: Start TabGAN training workflow using cached config
  - `get_workflow_logs`: Stream logs with configurable level (DEBUG/INFO/WARN/ERROR) and timeout
  - `get_trained_model_id`: Extract model ID from completed training workflow
- **Generation**:
  - `start_generation_workflow`: Start generation workflow from trained model
  - `obtain_synthetic_dataset_id`: Extract generated dataset ID from completed workflow
- **Quality Assessment**:
  - `plot_distribution`: Generate distribution plots (bar for categorical, KDE for numerical) comparing datasets
  - `get_marginal_distribution_score`: Calculate similarity score between real and synthetic data distributions

### Key Implementation Details

- All API calls are asynchronous:
  - HTTP clients (`rockfish_client`, `manta_client`) use `httpx.AsyncClient`
  - SDK client uses native async Rockfish SDK via `rf.Connection.remote()`
- All clients use a centralized `call_endpoint()` method with if/elif routing for tool dispatch
- Server initialization:
  - Always creates global `RockfishClient` and `RockfishSDKClient` instances
  - Only creates `MantaClient` instance if `MANTA_API_URL` environment variable is set
  - Manta tools are dynamically added to the tool list only when `manta_client` is initialized
- Tool routing is handled by checking:
  - SDK tools (in `sdk_tools` list) route to `sdk_client`
  - Manta tools (in `manta_v2_tools` list) route to `manta_client`
  - All other tools route to `rockfish_client`
- Tool schemas are defined inline using JSON Schema format directly in the server
- Error handling returns `types.TextContent` objects (or `ImageContent` for plots) for display to users
- Each tool specifies required fields and optional parameters in its input schema
- HTTP clients extract IDs and parameters from arguments and construct appropriate URL paths
- SDK client maintains in-memory cache for training configurations and returns structured error responses
- Manta tools require `organization_id` and `project_id` in every request (passed as headers)

All three clients abstract their respective complexities (REST API, Manta service, Rockfish SDK), while the server provides a unified MCP interface that AI assistants can use to interact with Rockfish resources, Manta testing capabilities, and synthetic data generation workflows programmatically.

## API Reference

For complete API documentation, see:
- **Rockfish API**: https://docs.rockfish.ai/openapi.yaml
- **Manta Service**: https://manta.rockfish.ai/openapi.json
