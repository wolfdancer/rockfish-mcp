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

Create a `.env` file with these variables for local development:
```bash
# Copy the example and edit with your values
cp .env.example .env
```

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides AI assistants access to the Rockfish machine learning platform API. The architecture consists of two main components in a simple, focused structure.

### Project Structure
```
src/rockfish_mcp/
├── __init__.py
├── server.py      # MCP server with tool definitions and routing
└── client.py      # HTTP client for Rockfish API calls
```

### Core Components

**Server (`server.py`)**: The main MCP server that:
- Defines 22 tools across 6 resource categories (Databases, Worker Sets, Workflows, Models, Projects, Datasets)  
- Handles tool registration via `@server.list_tools()` decorator
- Routes all tool calls through `@server.call_tool()` decorator to the Rockfish client
- Manages server initialization and stdio communication with MCP protocol
- Uses global `rockfish_client` instance initialized in `main()`
- Requires `ROCKFISH_API_KEY` environment variable to function

**Client (`client.py`)**: HTTP client wrapper that:
- Handles Bearer token authentication for all API requests
- Provides async HTTP requests to Rockfish API endpoints via httpx
- Maps MCP tool names to specific HTTP endpoints and methods in `call_endpoint()`
- Uses different HTTP methods (GET, POST, PUT, PATCH, DELETE) based on operation
- Centralizes error handling with `raise_for_status()` and returns formatted responses

### Tool Categories and API Mapping
The server exposes CRUD operations mapping to these endpoints:
- **Databases**: `/database` endpoints (GET, POST, PUT, DELETE)
- **Worker Sets**: `/worker-set` endpoints (GET, POST, DELETE - no update)  
- **Workflows**: `/workflow` endpoints (GET, POST, PUT)
- **Models**: `/models` endpoints (GET, POST, DELETE - note different path)
- **Projects**: `/project` endpoints (GET, POST, PATCH)
- **Datasets**: `/dataset` endpoints (GET, POST, PATCH, DELETE)

### Key Implementation Details

- All API calls are asynchronous using `httpx.AsyncClient` with proper connection handling
- The client uses a centralized `call_endpoint()` method with if/elif routing for tool dispatch
- Server initialization creates a single global `RockfishClient` instance shared across all calls
- Tool schemas are defined inline using JSON Schema format directly in the server
- Error handling returns `types.TextContent` objects for display to users
- Each tool specifies required fields and optional parameters in its input schema
- The client extracts IDs from arguments and constructs appropriate URL paths

The client abstracts the REST API complexity, while the server provides the MCP interface that AI assistants can use to interact with Rockfish resources programmatically.

## API Reference

For complete API documentation, see the OpenAPI specification at: https://docs.rockfish.ai/openapi.yaml