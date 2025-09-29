#!/usr/bin/env python3

import asyncio
import logging
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from .client import RockfishClient

load_dotenv()

logger = logging.getLogger("rockfish-mcp")

server = Server("rockfish-mcp")

# Global client instance
rockfish_client: Optional[RockfishClient] = None


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available Rockfish API tools."""
    tools = [
        # Database tools
        types.Tool(
            name="list_databases",
            description="List all databases",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_database",
            description="Create a new database",
            inputSchema={
                "type": "object", 
                "properties": {
                    "name": {"type": "string", "description": "Database name"},
                    "description": {"type": "string", "description": "Database description"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_database",
            description="Get a specific database by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_database",
            description="Update a database",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"},
                    "name": {"type": "string", "description": "Database name"},
                    "description": {"type": "string", "description": "Database description"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_database",
            description="Delete a database",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"}
                },
                "required": ["id"]
            }
        ),
        
        # Worker Set tools
        types.Tool(
            name="list_worker_sets",
            description="List all worker sets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_worker_set",
            description="Create a new worker set",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Worker set name"},
                    "worker_count": {"type": "integer", "description": "Number of workers"}
                },
                "required": ["name", "worker_count"]
            }
        ),
        types.Tool(
            name="get_worker_set",
            description="Get a specific worker set by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_worker_set",
            description="Delete a worker set",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="get_worker_set_actions",
            description="List the actions for a worker-set",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"]
            }
        ),
        
        # Workflow tools
        types.Tool(
            name="list_workflows",
            description="List all workflows",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_workflow",
            description="Create and run a new workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Workflow name"},
                    "definition": {"type": "object", "description": "Workflow definition"}
                },
                "required": ["name", "definition"]
            }
        ),
        types.Tool(
            name="get_workflow",
            description="Get a specific workflow by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_workflow",
            description="Update a workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID"},
                    "name": {"type": "string", "description": "Workflow name"},
                    "definition": {"type": "object", "description": "Workflow definition"}
                },
                "required": ["id"]
            }
        ),
        
        # Models tools
        types.Tool(
            name="list_models",
            description="List all models",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="upload_model",
            description="Upload a new model",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name"},
                    "metadata": {"type": "object", "description": "Model metadata"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_model",
            description="Get a specific model by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Model ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_model",
            description="Delete a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Model ID"}
                },
                "required": ["id"]
            }
        ),
        
        # Project tools
        types.Tool(
            name="list_projects",
            description="List all projects",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_project",
            description="Create a new project",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_project",
            description="Get a specific project by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Project ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_project",
            description="Update a project",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Project ID"},
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"}
                },
                "required": ["id"]
            }
        ),
        
        # Dataset tools
        types.Tool(
            name="list_datasets",
            description="List all datasets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_dataset",
            description="Create a new dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {"type": "string", "description": "Dataset description"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_dataset",
            description="Get a specific dataset by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_dataset",
            description="Update a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"},
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {"type": "string", "description": "Dataset description"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_dataset",
            description="Delete a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),
        
        # Query tools
        types.Tool(
            name="execute_query",
            description="Execute a query and return results in CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to execute"},
                    "project_id": {"type": "string", "description": "Optional project ID to execute the query in"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="query_dataset",
            description="Execute a query against a specific dataset and return results in CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID to query against"},
                    "query": {"type": "string", "description": "The query to execute"},
                    "project_id": {"type": "string", "description": "Optional project ID to execute the query in"}
                },
                "required": ["id", "query"]
            }
        )
    ]
    
    return tools


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool calls to Rockfish API."""
    
    if not rockfish_client:
        return [types.TextContent(
            type="text", 
            text="Rockfish client not initialized. Please check your API credentials."
        )]
    
    try:
        result = await rockfish_client.call_endpoint(name, arguments)
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        logger.error(f"Error calling {name}: {e}")
        return [types.TextContent(
            type="text", 
            text=f"Error calling {name}: {str(e)}"
        )]


async def main():
    global rockfish_client
    
    # Initialize Rockfish client
    api_key = os.getenv("ROCKFISH_API_KEY")
    base_url = os.getenv("ROCKFISH_BASE_URL", "https://api.rockfish.ai")
    
    if not api_key:
        logger.error("ROCKFISH_API_KEY environment variable is required")
        return
        
    rockfish_client = RockfishClient(api_key=api_key, base_url=base_url)
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rockfish-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def cli():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()