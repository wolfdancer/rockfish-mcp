# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install from requirements.txt (for development)
pip install -r requirements.txt
```

### Running the Server
```bash
# Run the MCP server
python -m rockfish_mcp.server

# Or use the console script
rockfish-mcp
```

### Environment Setup
The application requires these environment variables:
- `ROCKFISH_API_KEY`: Your Rockfish API key (required)
- `ROCKFISH_BASE_URL`: Base URL for Rockfish API (defaults to https://api.rockfish.ai)
- `ROCKFISH_ORGANIZATION_ID`: Organization ID (optional - uses default organization if not set)
- `ROCKFISH_PROJECT_ID`: Project ID (optional - uses default project if not set)
- `MANTA_BASE_URL`: Base URL for Manta service (optional - Manta tools only appear if this is set)

Create a `.env` file with these variables for local development:
```bash
# Copy the example and edit with your values
cp .env.example .env
```

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides AI assistants access to the Rockfish machine learning platform API and the Manta dataset testing service. The architecture consists of three main components in a simple, focused structure.

### Project Structure
```
src/rockfish_mcp/
├── __init__.py
├── server.py       # MCP server with tool definitions and routing
├── client.py       # HTTP client for Rockfish API calls
└── manta_client.py # HTTP client for Manta service calls
```

### Core Components

**Server (`server.py`)**: The main MCP server that:
- Defines tools across multiple resource categories
  - Rockfish API: Databases, Worker Sets, Workflows, Models, Organizations, Projects, Datasets, Queries (34 tools, always available)
  - Manta Service: Prompt Management, Data Manipulation, LLM Processing (10 tools, conditional)
- Conditionally loads Manta tools only when `MANTA_BASE_URL` environment variable is set
- Handles tool registration via `@server.list_tools()` decorator
- Routes tool calls through `@server.call_tool()` decorator:
  - Tools prefixed with `manta_` route to `manta_client`
  - All other tools route to `rockfish_client`
- Manages server initialization and stdio communication with MCP protocol
- Uses global `rockfish_client` (always) and `manta_client` (conditional) instances initialized in `main()`
- Requires `ROCKFISH_API_KEY` environment variable to function
- Supports optional `ROCKFISH_ORGANIZATION_ID` and `ROCKFISH_PROJECT_ID` environment variables for scoping API calls

**Client (`client.py`)**: HTTP client wrapper for Rockfish API that:
- Handles Bearer token authentication for all API requests
- Provides async HTTP requests to Rockfish API endpoints via httpx
- Maps MCP tool names to specific HTTP endpoints and methods in `call_endpoint()`
- Uses different HTTP methods (GET, POST, PUT, PATCH, DELETE) based on operation
- Supports optional `X-Organization-ID` and `X-Project-ID` headers for scoping API requests
- Centralizes error handling with `raise_for_status()` and returns formatted responses

**Manta Client (`manta_client.py`)**: HTTP client wrapper for Manta service that:
- Handles Bearer token authentication (uses same `ROCKFISH_API_KEY`)
- Provides async HTTP requests to Manta service endpoints via httpx
- Manages required Manta headers (`X-Organization-ID`, `X-Project-ID`)
- Maps Manta tool names to specific endpoints for:
  - Prompt management (create, get, append, evaluate)
  - Incident injection (spike, magnitude change, outage, ramp)
  - LLM question processing
- Centralizes error handling and returns formatted responses

### Tool Categories and API Mapping

#### Rockfish API Endpoints
The server exposes CRUD operations mapping to these endpoints:
- **Databases**: `/database` endpoints (GET, POST, PUT, DELETE)
- **Worker Sets**: `/worker-set` endpoints (GET, POST, DELETE - no update) and `/worker-set/{id}/actions`
- **Worker Groups**: `/worker-group` endpoint for listing available actions across all worker sets
- **Workflows**: `/workflow` endpoints (GET, POST, PUT)
- **Models**: `/models` endpoints (GET, POST, DELETE - note different path)
- **Organizations**: `/organization` endpoints (GET for list, GET `/organization/active` for active org)
- **Projects**: `/project` endpoints (GET, POST, PATCH, and GET `/project/active` for active project)
- **Datasets**: `/dataset` endpoints (GET, POST, PATCH, DELETE, and GET `/dataset/{id}/schema` for schema)
- **Queries**: `/query` endpoint (POST with text/plain query) and `/dataset/{id}/query` for dataset-specific queries

#### Manta Service Endpoints (Optional)
The Manta service provides dataset testing and pattern injection capabilities. These tools are only available when `MANTA_BASE_URL` is configured:
- **Prompt Management**: `/prompts` endpoints (GET, POST, PATCH)
  - Generate and manage test prompts for datasets
  - Evaluate test case results
- **Incident Injection**: Data manipulation endpoints
  - `/instantaneous-spike-data`: Inject sudden spikes
  - `/sustained-magnitude-change-data`: Apply sustained changes
  - `/data-outage-data`: Create data gaps
  - `/value-ramp-data`: Apply gradual changes
  - `/incident-dataset-ids`: List all incident datasets
- **LLM Processing**: `/customer-llm` endpoint
  - Process natural language questions via SQL Agent

### Key Implementation Details

- All API calls are asynchronous using `httpx.AsyncClient` with proper connection handling
- Both clients use a centralized `call_endpoint()` method with if/elif routing for tool dispatch
- Server initialization:
  - Always creates a global `RockfishClient` instance
  - Only creates `MantaClient` instance if `MANTA_BASE_URL` environment variable is set
  - Manta tools are dynamically added to the tool list only when `manta_client` is initialized
  - Optional `ROCKFISH_ORGANIZATION_ID` and `ROCKFISH_PROJECT_ID` are passed to RockfishClient for scoping
- Tool routing is handled by checking tool name prefix (`manta_` routes to Manta, others to Rockfish)
- Tool schemas are defined inline using JSON Schema format directly in the server
- Error handling returns `types.TextContent` objects for display to users
- Each tool specifies required fields and optional parameters in its input schema
- The clients extract IDs and parameters from arguments and construct appropriate URL paths
- Manta tools require `organization_id` and `project_id` in every request (passed as headers)
- Query tools (`execute_query`, `query_dataset`) use special handling:
  - Send queries as `text/plain` content instead of JSON
  - Support optional `X-Project-ID` header for scoping queries
  - Return results in CSV format from the API
- Organization and Project headers (`X-Organization-ID`, `X-Project-ID`) scope API requests when provided

Both clients abstract REST API complexity, while the server provides a unified MCP interface that AI assistants can use to interact with Rockfish resources and Manta testing capabilities programmatically.

## API Reference

For complete API documentation, see:
- **Rockfish API**: https://docs.rockfish.ai/openapi.yaml
- **Manta Service**: https://manta.sunset-beach.rockfish.ai/openapi.json

## Tool Categories Reference

### Database Tools (5)
- `list_databases`, `create_database`, `get_database`, `update_database`, `delete_database`

### Worker Set Tools (5)
- `list_worker_sets`, `create_worker_set`, `get_worker_set`, `delete_worker_set`, `get_worker_set_actions`

### Worker Group Tools (1)
- `list_available_actions` - Lists all actions across all worker sets

### Workflow Tools (4)
- `list_workflows`, `create_workflow`, `get_workflow`, `update_workflow`

### Model Tools (4)
- `list_models`, `upload_model`, `get_model`, `delete_model`

### Organization Tools (2)
- `get_active_organization`, `list_organizations`

### Project Tools (5)
- `get_active_project`, `list_projects`, `create_project`, `get_project`, `update_project`

### Dataset Tools (5)
- `list_datasets`, `create_dataset`, `get_dataset`, `update_dataset`, `delete_dataset`

### Dataset Schema Tools (1)
- `get_dataset_schema` - Get metadata and schema information for a dataset

### Query Tools (2)
- `execute_query` - Execute a query and return CSV results (optional project_id parameter)
- `query_dataset` - Execute a query against a specific dataset and return CSV results

### Manta Tools (10 - conditional, only available if MANTA_BASE_URL is set)
#### Prompt Management (4)
- `manta_get_prompts`, `manta_create_prompts`, `manta_append_prompts`, `manta_evaluate_test_case`

#### Incident Injection (5)
- `manta_create_instantaneous_spike`, `manta_create_sustained_magnitude_change`
- `manta_create_data_outage`, `manta_create_value_ramp`, `manta_get_incident_dataset_ids`

#### LLM Processing (1)
- `manta_process_llm_questions` - Process natural language questions using SQL Agent

## Development Guidelines for AI Assistants

### Working with Code
1. **Always read files before modifying**: Use the Read tool to understand the current implementation before making changes
2. **Maintain consistency**: Follow the existing patterns in the codebase:
   - Tool definitions in `server.py` use JSON Schema for input validation
   - Route handling in `call_endpoint()` methods use if/elif chains
   - All API calls are async and use `httpx.AsyncClient`
3. **Error handling**: Always use `raise_for_status()` and return `types.TextContent` for errors
4. **Avoid over-engineering**: Keep the simple, focused structure - don't add unnecessary abstractions

### Adding New Tools
When adding a new tool to the MCP server:
1. **Define the tool** in `server.py` in the `handle_list_tools()` function
   - Use descriptive name and description
   - Define complete JSON Schema for inputs
   - Mark required vs optional parameters
2. **Add routing logic** in the appropriate client:
   - Rockfish tools go in `client.py` → `call_endpoint()`
   - Manta tools go in `manta_client.py` → `call_endpoint()`
   - Prefix Manta tools with `manta_` in the name
3. **Handle the endpoint** in the client's `call_endpoint()` method
   - Extract IDs and parameters from arguments
   - Use appropriate HTTP method (GET, POST, PUT, PATCH, DELETE)
   - Construct correct URL path
   - Return response JSON or handle text responses appropriately
4. **Update documentation** in CLAUDE.md and README.md with the new tool

### Query Tool Usage Patterns
- Use `execute_query` for general queries across the project
- Use `query_dataset` when querying a specific dataset
- Both return results in CSV format
- Optional `project_id` parameter can scope queries to specific projects
- Queries are sent as `text/plain` content, not JSON

### Organization and Project Scoping
- Set `ROCKFISH_ORGANIZATION_ID` and `ROCKFISH_PROJECT_ID` environment variables to scope all API calls
- These are passed as headers (`X-Organization-ID`, `X-Project-ID`) to the Rockfish API
- If not set, the API will use default organization/project for the authenticated user
- Manta tools always require explicit `organization_id` and `project_id` parameters

### Testing Changes
1. **Local testing**: Run the server with `python -m rockfish_mcp.server` and verify no startup errors
2. **MCP Inspector**: Use `npx @modelcontextprotocol/inspector` to test individual tools
3. **Environment**: Ensure `.env` file has all required variables set for testing
4. **Tool validation**: Test both success and error cases for new/modified tools

### Python Version Compatibility
- Target Python >=3.8 as specified in `pyproject.toml`
- Avoid using Python 3.13+ features
- Use type hints compatible with Python 3.8 (from `typing` module)
- Test with Python 3.11 or 3.12 for best compatibility