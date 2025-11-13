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
- `MANTA_BASE_URL`: Base URL for Manta service (optional - Manta tools only appear if this is set)

Create a `.env` file with these variables for local development:
```bash
# Copy the example and edit with your values
cp .env.example .env
```

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides AI assistants access to the Rockfish machine learning platform API and the Manta dataset testing service. The architecture uses a **hybrid approach**: the official Rockfish Python SDK for most operations (providing better type safety and future-proofing), with HTTP/REST API fallback for operations not supported by the SDK.

### Project Structure
```
src/rockfish_mcp/
├── __init__.py
├── server.py       # MCP server with tool definitions and intelligent routing
├── sdk_client.py   # SDK client using official Rockfish Python SDK (primary)
├── client.py       # HTTP client for SDK-unsupported operations (fallback)
└── manta_client.py # HTTP client for Manta service calls
```

### Core Components

**Server (`server.py`)**: The main MCP server that:
- Defines tools across multiple resource categories
  - Rockfish API: Databases, Worker Sets, Workflows, Models, Projects, Datasets (22 tools, always available)
  - Manta Service: Prompt Management, Data Manipulation, LLM Processing (10 tools, conditional)
- Conditionally loads Manta tools only when `MANTA_BASE_URL` environment variable is set
- Handles tool registration via `@server.list_tools()` decorator
- Routes tool calls through `@server.call_tool()` decorator with intelligent routing:
  - Tools prefixed with `manta_` → `manta_client`
  - HTTP-only tools (databases, worker-sets, certain dataset/model ops) → `http_client`
  - All other Rockfish tools → `sdk_client` (with HTTP fallback if needed)
- Manages server initialization and stdio communication with MCP protocol
- Uses global `sdk_client`, `http_client`, and `manta_client` instances initialized in `main()`
- Requires `ROCKFISH_API_KEY` environment variable to function

**SDK Client (`sdk_client.py`)**: Official Rockfish SDK wrapper (primary client) that:
- Uses the official `rockfish` Python SDK package for better type safety
- Initializes via `Connection.from_env()` to automatically read environment variables
- Provides async SDK operations for: organizations, projects, workflows, datasets (list/get/delete/query), models (list/get)
- Returns dictionaries converted from SDK objects (Project, Dataset, Model, etc.)
- Handles Stream objects for paginated listing operations
- Raises `NotImplementedError` for operations not supported by SDK (triggers HTTP fallback)
- **Benefits**: Type safety, future-proof, better error handling, maintained by Rockfish team

**HTTP Client (`client.py`)**: HTTP/REST client for SDK-unsupported operations (fallback) that:
- Handles operations not easily supported by the SDK: databases, worker-sets, dataset create/update/schema, model upload/delete
- Provides async HTTP requests to Rockfish API endpoints via httpx
- Maps MCP tool names to specific HTTP endpoints and methods in `call_endpoint()`
- Uses different HTTP methods (GET, POST, PUT, PATCH, DELETE) based on operation
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
- **Worker Sets**: `/worker-set` endpoints (GET, POST, DELETE - no update)
- **Workflows**: `/workflow` endpoints (GET, POST, PUT)
- **Models**: `/models` endpoints (GET, POST, DELETE - note different path)
- **Projects**: `/project` endpoints (GET, POST, PATCH)
- **Datasets**: `/dataset` endpoints (GET, POST, PATCH, DELETE)

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

**Hybrid SDK + HTTP Architecture:**
- **SDK Client (primary)**: Uses official `rockfish` package via `Connection.from_env()`
  - Handles 18 operations (~56%): organizations, projects, workflows, datasets (partial), models (partial)
  - Provides type safety, better error handling, and automatic SDK updates
  - Returns dictionaries converted from SDK objects for MCP compatibility
- **HTTP Client (fallback)**: Uses `httpx.AsyncClient` for direct REST API calls
  - Handles 14 operations (~44%): databases, worker-sets, dataset CRUD, model upload/delete
  - Required for operations not supported by SDK or requiring complex object construction
- **Manta Client**: Separate HTTP client for Manta service (10 tools)

**Routing Logic:**
1. **Manta tools** (`manta_*` prefix) → route to `manta_client`
2. **HTTP-only tools** (databases, worker-sets, etc.) → route to `http_client`
3. **All other Rockfish tools** → route to `sdk_client` with automatic HTTP fallback on `NotImplementedError`

**Server Initialization:**
- Always creates `sdk_client` (using `Connection.from_env()`) and `http_client` instances
- Only creates `manta_client` if `MANTA_BASE_URL` environment variable is set
- Manta tools are dynamically added to the tool list only when configured

**General:**
- All API calls are asynchronous with proper connection handling
- All clients use a centralized `call_endpoint()` method with if/elif routing for tool dispatch
- Tool schemas are defined inline using JSON Schema format directly in the server
- Error handling returns `types.TextContent` objects for display to users
- Each tool specifies required fields and optional parameters in its input schema
- Manta tools require `organization_id` and `project_id` in every request (passed as headers)

The hybrid architecture maximizes use of the official SDK while maintaining full API coverage through HTTP fallback, providing a unified MCP interface for AI assistants to interact with Rockfish resources programmatically.

## API Reference

For complete API documentation, see:
- **Rockfish API**: https://docs.rockfish.ai/openapi.yaml
- **Manta Service**: https://manta.sunset-beach.rockfish.ai/openapi.json