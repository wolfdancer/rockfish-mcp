#!/usr/bin/env python3

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from .client import RockfishHTTPClient
from .sdk_client import RockfishSDKClient
from .manta_client import MantaClient
from .recommender_client import RecommenderClient
from .sda_client import SDAClient
from .sdk_docstring_extractor import get_extractor

load_dotenv()

logger = logging.getLogger("rockfish-mcp")

server = Server("rockfish-mcp")

# Global client instances
http_client: Optional[RockfishHTTPClient] = None  # HTTP/REST API (fallback)
sdk_client: Optional[RockfishSDKClient] = None     # Official SDK (primary)
manta_client: Optional[MantaClient] = None         # Manta service
recommender_client: Optional[RecommenderClient] = None  # Recommender service
sda_client: Optional[SDAClient] = None             # SDA (Synthetic Data Assessment)

# Initialize SDK docstring extractor
extractor = get_extractor()


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available Rockfish API, Manta service, and Recommender service tools."""
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

        # Worker tools
        types.Tool(
            name="list_available_actions",
            description="List actions (or workers) from all worker-sets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        # Workflow tools
        types.Tool(
            name="list_workflows",
            description=extractor.format_tool_description(
                "List workflows in the active project",
                "workflows"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_workflow",
            description=extractor.format_tool_description(
                "Create and execute a new workflow",
                "create_workflow"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                    "description": "Alias for this job (used for referencing in inputs)"
                                },
                                "config": {
                                    "type": "object",
                                    "description": "Job configuration (can contain any action config)"
                                },
                                "input": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "List of input aliases this job depends on"
                                },
                                "replicas": {
                                    "type": "integer",
                                    "default": 1,
                                    "description": "Number of replicas for this job"
                                },
                                "worker_name": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                    "description": "Name of the worker to execute this job"
                                },
                                "worker_version": {
                                    "type": "string",
                                    "description": "Version of the worker"
                                }
                            },
                            "required": ["alias", "config", "input", "worker_name", "worker_version"]
                        },
                        "minItems": 1,
                        "description": "Array of jobs that make up the workflow"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                "description": "Workflow name"
                            },
                            "labels": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN _-]{1,64}$"
                                },
                                "description": "Key-value labels for the workflow"
                            }
                        },
                        "required": ["name", "labels"],
                        "description": "Workflow metadata"
                    }
                },
                "required": ["jobs", "metadata"],
                "description": "Workflow definition with jobs and metadata"
            }
        ),
        types.Tool(
            name="get_workflow",
            description=extractor.format_tool_description(
                "Get a specific workflow by ID",
                "get_workflow"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="visualize_workflow",
            description="Generate a Mermaid diagram visualization of a workflow showing job structure and status. "
            "Returns a Mermaid markdown diagram with color-coded job states (blue=created, yellow=started, green=success, red=failure).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID to visualize"},
                    "direction": {
                        "type": "string",
                        "enum": ["LR", "TB"],
                        "default": "LR",
                        "description": "Graph direction: 'LR' for left-to-right (horizontal), 'TB' for top-to-bottom (vertical)"
                    }
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="visualize_workflow_builder",
            description="Build and visualize a workflow structure WITHOUT executing it. "
            "Generates a Mermaid diagram showing the planned workflow DAG structure using WorkflowBuilder. "
            "Perfect for previewing, designing, and validating workflow architecture before execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "Array of action configurations to build the workflow",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_class": {
                                    "type": "string",
                                    "description": "SDK action class name (e.g., 'DatasetLoad', 'TabPropertyExtractor', 'Generate')"
                                },
                                "config": {
                                    "type": "object",
                                    "description": "Configuration parameters for the action"
                                },
                                "alias": {
                                    "type": "string",
                                    "description": "Optional alias for the action in the diagram"
                                }
                            },
                            "required": ["action_class"]
                        },
                        "minItems": 1
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["LR", "TB"],
                        "default": "LR",
                        "description": "Graph direction: 'LR' for left-to-right (horizontal), 'TB' for top-to-bottom (vertical)"
                    }
                },
                "required": ["actions"]
            }
        ),
        types.Tool(
            name="plot_dataset_distribution",
            description="Generate a distribution plot for a column in a dataset. "
            "Returns a PNG image visualization showing the distribution (histogram for numerical data, bar chart for categorical data). "
            "Uses the Rockfish SDK's rockfish.labs.vis.plot_distribution() function.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to plot"},
                    "column_name": {"type": "string", "description": "Column name to visualize"},
                    "bins": {"type": "integer", "description": "Number of bins for histogram (numerical data only)", "default": 30}
                },
                "required": ["dataset_id", "column_name"]
            }
        ),
        types.Tool(
            name="update_workflow",
            description=extractor.format_tool_description(
                "Update an existing workflow",
                "update_workflow"
            ),
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
            description=extractor.format_tool_description(
                "List all models in the active project",
                "models"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="upload_model",
            description="Upload a new model\n\n**Note:** This operation uses HTTP API, not SDK.",
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
            description=extractor.format_tool_description(
                "Get a specific model by ID",
                "get_model"
            ),
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

        # Organization tools
        types.Tool(
            name="get_active_organization",
            description=extractor.format_tool_description(
                "Get the currently active organization",
                "active_organization"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="list_organizations",
            description=extractor.format_tool_description(
                "List all organizations you have access to",
                "organizations"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        # Project tools
        types.Tool(
            name="get_active_project",
            description=extractor.format_tool_description(
                "Get the currently active project",
                "active_project"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="list_projects",
            description=extractor.format_tool_description(
                "List all projects in the active organization",
                "projects"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_project",
            description=extractor.format_tool_description(
                "Create a new project in the active organization",
                "create_project"
            ),
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
            description=extractor.format_tool_description(
                "Get a specific project by ID",
                "projects"
            ) + "\n\n**Note:** SDK implementation lists and filters projects.",
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
            description=extractor.format_tool_description(
                "List all datasets in the active project",
                "datasets"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_dataset",
            description="Create a new dataset\n\n**Note:** This operation uses HTTP API, not SDK.",
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
            description=extractor.format_tool_description(
                "Get a specific dataset by ID",
                "get_dataset"
            ),
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
            description="Update a dataset\n\n**Note:** This operation uses HTTP API, not SDK.",
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
            description=extractor.format_tool_description(
                "Delete a dataset by ID",
                "delete_dataset"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),

        # Dataset schema tools
        types.Tool(
            name="get_dataset_schema",
            description="Get the schema for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),

        # Dataset properties extraction (SDK workflow-based)
        types.Tool(
            name="extract_tabular_properties",
            description="Extract tabular dataset properties using SDK workflows (PII detection, association rules, field types). "
            "Creates a workflow that loads the dataset, extracts properties, and returns both dataset-level and field-level properties.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to analyze"
                    },
                    "detect_pii": {
                        "type": "boolean",
                        "description": "Whether to detect personally identifiable information (PII)",
                        "default": False
                    },
                    "detect_association_rules": {
                        "type": "boolean",
                        "description": "Whether to detect field association rules",
                        "default": False
                    },
                    "association_threshold": {
                        "type": "number",
                        "description": "Threshold for association rule detection (0-1)",
                        "default": 0.95
                    }
                },
                "required": ["dataset_id"]
            }
        ),
        types.Tool(
            name="extract_timeseries_properties",
            description="Extract timeseries dataset properties using SDK workflows (PII detection, association rules, field types, session analysis). "
            "Creates a workflow that loads the dataset, extracts properties, and returns both dataset-level and field-level properties.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to analyze"
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "Name of the timestamp field in the timeseries dataset (required)"
                    },
                    "session_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of fields that define sessions/entities (e.g., ['user_id', 'device_id'])",
                        "default": []
                    },
                    "metadata_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of fields to treat as metadata. Leave unset to use auto-detection.",
                        "default": None
                    },
                    "detect_metadata_fields": {
                        "type": "boolean",
                        "description": "Whether to automatically detect metadata fields",
                        "default": False
                    },
                    "detect_pii": {
                        "type": "boolean",
                        "description": "Whether to detect personally identifiable information (PII)",
                        "default": False
                    },
                    "detect_association_rules": {
                        "type": "boolean",
                        "description": "Whether to detect field association rules",
                        "default": False
                    },
                    "association_threshold": {
                        "type": "number",
                        "description": "Threshold for association rule detection (0-1)",
                        "default": 0.95
                    }
                },
                "required": ["dataset_id", "timestamp"]
            }
        ),

        # Query tools
        types.Tool(
            name="execute_query",
            description=extractor.format_tool_description(
                "Execute a SQL query across datasets and return results",
                "query_datasets"
            ),
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
            description=extractor.format_tool_description(
                "Execute a SQL query against a specific dataset",
                "query_dataset"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID to query against"},
                    "query": {"type": "string", "description": "The query to execute"},
                    "project_id": {"type": "string", "description": "Optional project ID to execute the query in"}
                },
                "required": ["id", "query"]
            }
        ),

        # SDK Documentation tools
        types.Tool(
            name="get_sdk_docs",
            description="Get Rockfish SDK documentation for Connection class methods. "
            "Use this to explore available SDK methods and their parameters. "
            "If method_name is not provided, lists all available methods.",
            inputSchema={
                "type": "object",
                "properties": {
                    "method_name": {
                        "type": "string",
                        "description": "Optional: specific method name (e.g., 'datasets', 'create_project'). "
                        "If omitted, returns a list of all available methods."
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="list_sdk_actions",
            description="List all 63+ Rockfish SDK action classes with brief descriptions. "
            "Actions are building blocks for workflows (e.g., DatasetLoad, TabPropertyExtractor, Generate). "
            "Use this to discover available actions for building workflows.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_action_docs",
            description="Get detailed documentation for a specific Rockfish SDK action class. "
            "Returns full docstring with usage examples, parameter descriptions, and configuration schema. "
            "Use list_sdk_actions first to see available action names.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_name": {
                        "type": "string",
                        "description": "Action class name (e.g., 'TabPropertyExtractor', 'DatasetLoad', 'Generate')"
                    }
                },
                "required": ["action_name"]
            }
        ),
        types.Tool(
            name="get_action_config_schema",
            description="Get the JSON configuration schema for a specific Rockfish SDK action class. "
            "Returns the JSON schema showing required and optional parameters, types, defaults, and constraints. "
            "This is useful for understanding what configuration parameters an action accepts when building workflows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_name": {
                        "type": "string",
                        "description": "Action class name (e.g., 'TabPropertyExtractor', 'DatasetLoad', 'SQL')"
                    }
                },
                "required": ["action_name"]
            }
        ),

        # SDA (Synthetic Data Assessment) tools
        types.Tool(
            name="generate_quality_report",
            description="Generate a comprehensive data quality assessment report comparing real and synthetic datasets. "
            "Uses the SDA (Synthetic Data Assessment) module to analyze data quality, detect patterns, "
            "and generate an HTML report. All parameters are optional with smart defaults.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the real dataset to assess (default: '33jhtc7JwDPA7jRwvfPrwI')"
                    },
                    "syn_id": {
                        "type": "string",
                        "description": "ID of the synthetic dataset to compare against (default: '56rWfb2iWU1jtcu0oCimD6')"
                    },
                    "config": {
                        "type": "object",
                        "description": "Configuration dict with 'encoder' and 'tabular-gan' settings. "
                        "If not provided, will auto-generate by detecting categorical fields from the dataset.",
                        "properties": {
                            "encoder": {
                                "type": "object",
                                "properties": {
                                    "metadata": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "field": {"type": "string"},
                                                "type": {"type": "string", "enum": ["categorical", "continuous"]}
                                            }
                                        }
                                    }
                                }
                            },
                            "tabular-gan": {
                                "type": "object",
                                "properties": {
                                    "epochs": {"type": "integer"},
                                    "records": {"type": "integer"}
                                }
                            }
                        }
                    },
                    "output_file": {
                        "type": "string",
                        "description": "HTML output filename (default: 'test1.html'). Report saved to ~/rockfish-reports/ directory (or temp directory as fallback)."
                    }
                },
                "required": []
            }
        )
    ]

    # Add Manta tools only if Manta client is initialized
    if manta_client:
        manta_tools = [
            # Manta Service - Prompt Management tools
            types.Tool(
            name="manta_get_prompts",
            description="Get natural language prompts and expected answers for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to get prompts for"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_prompts",
            description="Create prompts and answers for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to create prompts for"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_append_prompts",
            description="Append prompts to an existing dataset prompt collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to append prompts to"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_evaluate_test_case",
            description="Compare actual vs expected results for test validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt/question that was asked"},
                    "actual_result": {"type": "string", "description": "The actual result obtained"},
                    "expected_result": {"type": "string", "description": "The expected result"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["prompt", "actual_result", "expected_result", "organization_id", "project_id"]
            }
        ),

        # Manta Service - Data Manipulation tools (Incident Injection)
        types.Tool(
            name="manta_create_instantaneous_spike",
            description="Create a modified dataset with an instantaneous spike incident injected",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the spike incident",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to inject spike into"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when spike occurs"},
                            "magnitude": {"type": "number", "description": "Magnitude of the spike"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_sustained_magnitude_change",
            description="Create a modified dataset with a sustained magnitude change incident",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the sustained magnitude change",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to modify"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when change starts"},
                            "incident_end_timestamp": {"type": "string", "description": "ISO timestamp when change ends"},
                            "magnitude": {"type": "number", "description": "Magnitude of the change"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_data_outage",
            description="Create a modified dataset with a data outage incident (missing data)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the data outage",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to create outage for"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when outage starts"},
                            "incident_end_timestamp": {"type": "string", "description": "ISO timestamp when outage ends"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_value_ramp",
            description="Create a modified dataset with a value ramp incident (gradual increase/decrease)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the value ramp",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to apply ramp to"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when ramp starts"},
                            "incident_end_timestamp": {"type": "string", "description": "ISO timestamp when ramp ends"},
                            "magnitude": {"type": "number", "description": "Total magnitude change over the ramp period"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_get_incident_dataset_ids",
            description="Get all incident dataset IDs associated with an original dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),

        # Manta Service - LLM Processing tools
        types.Tool(
            name="manta_process_llm_questions",
            description="Process natural language questions using SQL Agent functionality against a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to query"},
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of natural language questions to process"
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "questions", "organization_id", "project_id"]
            }
        )
        ]
        tools.extend(manta_tools)

    # Recommender tools (always available - uses main Rockfish API)
    recommender_tools = [
        types.Tool(
            name="recommender_generate_workflow",
            description="Generate a training workflow for a given model",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name for the generated dataset"},
                    "model_id": {"type": "string", "description": "ID of the model to use for generation"},
                    "sample_size": {"type": "integer", "description": "Number of samples to generate"},
                    "sample_matching_sql_query": {"type": "string", "description": "Optional SQL query to match sample distribution"}
                },
                "required": ["dataset_name", "model_id", "sample_size"]
            }
        ),
        types.Tool(
            name="recommender_evaluate_workflow",
            description="Generate an evaluation workflow for datasets and metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {"type": "array", "items": {"type": "string"}, "description": "List of dataset IDs to evaluate"},
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric_name": {"type": "string"},
                                "metric_config": {"type": "object"}
                            }
                        },
                        "description": "Optional list of metrics to compute"
                    }
                },
                "required": ["dataset_ids"]
            }
        ),
        types.Tool(
            name="recommender_train_workflow",
            description="Generate a training workflow for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID of the dataset to train on"}
                },
                "required": ["dataset_id"]
            }
        ),
        types.Tool(
            name="recommender_concat_workflow",
            description="Generate a concatenation workflow for multiple datasets",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {"type": "array", "items": {"type": "string"}, "description": "List of dataset IDs to concatenate"},
                    "dataset_name": {"type": "string", "description": "Name for the concatenated dataset"}
                },
                "required": ["dataset_ids", "dataset_name"]
            }
        ),
        types.Tool(
            name="recommender_tabular_properties",
            description="Detect and analyze properties of a tabular dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID of the tabular dataset to analyze"},
                    "detect_pii": {"type": "boolean", "description": "Whether to detect PII", "default": False},
                    "detect_association_rules": {"type": "boolean", "description": "Whether to detect field associations", "default": False},
                    "association_threshold": {"type": "number", "description": "Threshold for association rule detection (0-1)", "default": 0.95}
                },
                "required": ["dataset_id"]
            }
        ),
        types.Tool(
            name="recommender_timeseries_properties",
            description="Detect and analyze properties of a timeseries dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID of the timeseries dataset to analyze"},
                    "timestamp": {"type": "string", "description": "Name of the timestamp field"},
                    "session_fields": {"type": "array", "items": {"type": "string"}, "description": "Fields that define sessions", "default": []},
                    "metadata_fields": {"type": "array", "items": {"type": "string"}, "description": "Fields that contain metadata", "default": []},
                    "detect_pii": {"type": "boolean", "description": "Whether to detect PII", "default": False},
                    "detect_metadata_fields": {"type": "boolean", "description": "Whether to auto-detect metadata fields", "default": False},
                    "detect_association_rules": {"type": "boolean", "description": "Whether to detect field associations", "default": False},
                    "association_threshold": {"type": "number", "description": "Threshold for association rule detection (0-1)", "default": 0.95}
                },
                "required": ["dataset_id", "timestamp"]
            }
        ),
        types.Tool(
            name="recommender_dataset_fidelity_score",
            description="Calculate fidelity scores between datasets using SQL queries (minimum 2 datasets required)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "description": "List of dataset IDs to compare (minimum 2)"
                    }
                },
                "required": ["dataset_ids"]
            }
        ),
        types.Tool(
            name="recommender_sql_fidelity_checks",
            description="Get recommended SQL queries for fidelity checking",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {"type": "array", "items": {"type": "string"}, "description": "List of dataset IDs to generate checks for"}
                },
                "required": ["dataset_ids"]
            }
        ),
        types.Tool(
            name="recommender_generate_sources",
            description="Generate data generation sources from natural language prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Natural language description of what data you want to generate"}
                },
                "required": ["prompt"]
            }
        )
    ]
    tools.extend(recommender_tools)

    return tools


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent | types.ImageContent]:
    """Handle tool calls - routes to appropriate client (Manta, Recommender, SDK, or HTTP)."""

    # Route 1: Manta tools
    if name.startswith("manta_"):
        if not manta_client:
            return [types.TextContent(
                type="text",
                text="Manta client not initialized. Check MANTA_API_URL environment variable."
            )]

        try:
            result = await manta_client.call_endpoint(name, arguments)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Manta error calling {name}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error calling {name}: {str(e)}"
            )]

    # Route 2: Recommender tools
    if name.startswith("recommender_"):
        if not recommender_client:
            return [types.TextContent(
                type="text",
                text="Recommender client not initialized. Check ROCKFISH_PROJECT_ID and ROCKFISH_ORGANIZATION_ID."
            )]

        try:
            result = await recommender_client.call_endpoint(name, arguments)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Recommender error calling {name}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error calling {name}: {str(e)}"
            )]

    # Route 3: SDA tools (Synthetic Data Assessment)
    if name == "generate_quality_report":
        if not sda_client:
            return [types.TextContent(
                type="text",
                text="SDA client not initialized. Please check your configuration."
            )]

        try:
            result = await sda_client.generate_report(**arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.error(f"SDA error calling {name}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error calling {name}: {str(e)}"
            )]

    # Route 4: HTTP-only tools (operations SDK cannot handle)
    http_only_tools = [
        # Databases (5)
        "list_databases", "create_database", "get_database",
        "update_database", "delete_database",

        # Worker Sets (6)
        "list_worker_sets", "create_worker_set", "get_worker_set",
        "delete_worker_set", "get_worker_set_actions", "list_available_actions",

        # Dataset operations requiring complex objects (3)
        "create_dataset", "update_dataset", "get_dataset_schema",

        # Model operations requiring complex objects (2)
        "upload_model", "delete_model",

        # Project update (1)
        "update_project"
    ]

    if name in http_only_tools:
        if not http_client:
            return [types.TextContent(
                type="text",
                text="HTTP client not initialized. Please check your API credentials."
            )]

        try:
            result = await http_client.call_endpoint(name, arguments)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"HTTP error calling {name}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error calling {name}: {str(e)}"
            )]

    # Route 5: SDK tools (all other Rockfish operations - preferred)
    if not sdk_client:
        return [types.TextContent(
            type="text",
            text="SDK client not initialized. Please check your API credentials."
        )]

    try:
        result = await sdk_client.call_endpoint(name, arguments)

        # Check if result contains image data
        if isinstance(result, dict) and "image" in result and "mimeType" in result:
            return [types.ImageContent(
                type="image",
                data=result["image"],
                mimeType=result["mimeType"]
            )]

        # Default: return as text
        return [types.TextContent(type="text", text=str(result))]
    except NotImplementedError as e:
        # Fallback to HTTP client if SDK doesn't support the operation
        logger.warning(f"SDK doesn't support {name}, falling back to HTTP client")
        if http_client:
            try:
                result = await http_client.call_endpoint(name, arguments)
                return [types.TextContent(type="text", text=str(result))]
            except Exception as http_error:
                logger.error(f"HTTP fallback error calling {name}: {http_error}")
                return [types.TextContent(
                    type="text",
                    text=f"Error calling {name}: {str(http_error)}"
                )]
        return [types.TextContent(
            type="text",
            text=f"Operation not supported: {str(e)}"
        )]
    except Exception as e:
        logger.error(f"SDK error calling {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error calling {name}: {str(e)}"
        )]


async def main():
    global http_client, sdk_client, manta_client, recommender_client, sda_client

    # Check for required API key
    api_key = os.getenv("ROCKFISH_API_KEY")
    if not api_key:
        logger.error("ROCKFISH_API_KEY environment variable is required")
        return

    # Initialize SDK client (primary) - uses Connection.from_env()
    sdk_client = RockfishSDKClient()
    logger.info("SDK client initialized from environment")

    # Initialize SDA client (uses SDK connection)
    sda_client = SDAClient()
    logger.info("SDA client initialized")

    # Initialize HTTP client (fallback for SDK-unsupported operations)
    api_url = os.getenv("ROCKFISH_API_URL", "https://api.rockfish.ai")
    organization_id = os.getenv("ROCKFISH_ORGANIZATION_ID", None)
    project_id = os.getenv("ROCKFISH_PROJECT_ID", None)

    http_client = RockfishHTTPClient(
        api_key=api_key,
        api_url=api_url,
        organization_id=organization_id,
        project_id=project_id
    )
    logger.info("HTTP client initialized")

    # Initialize Recommender client (can be customized with RECOMMENDER_API_URL)
    recommender_api_url = os.getenv("RECOMMENDER_API_URL")
    if recommender_api_url:
        recommender_client = RecommenderClient(
            api_key=api_key,
            api_url=recommender_api_url,
            organization_id=organization_id,
            project_id=project_id
        )
        logger.info(f"Recommender client initialized with custom URL: {recommender_api_url}")
    else:
        recommender_client = RecommenderClient(
            api_key=api_key,
            organization_id=organization_id,
            project_id=project_id
        )
        logger.info("Recommender client initialized with default URL")

    # Initialize Manta client only if MANTA_API_URL is configured
    manta_api_url = os.getenv("MANTA_API_URL")
    if manta_api_url:
        manta_client = MantaClient(
            api_key=api_key,
            api_url=manta_api_url
        )
        logger.info(f"Manta client initialized with base URL: {manta_api_url}")
    else:
        logger.info("Manta client not initialized (MANTA_API_URL not set)")
    
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