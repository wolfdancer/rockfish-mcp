#!/usr/bin/env python3

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from dotenv import load_dotenv
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .client import RockfishClient
from .manta_client import MantaClient
from .sdk_client import RockfishSDKClient

load_dotenv()

logger = logging.getLogger("rockfish-mcp")

server = Server("rockfish-mcp")

# Global client instances
rockfish_client: Optional[RockfishClient] = None
manta_client: Optional[MantaClient] = None
sdk_client: Optional[RockfishSDKClient] = None


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available Rockfish API and Manta service tools."""
    tools = [
        # Database tools
        types.Tool(
            name="list_databases",
            description="List all databases",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="create_database",
            description="Create a new database",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Database name"},
                    "description": {
                        "type": "string",
                        "description": "Database description",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_database",
            description="Get a specific database by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Database ID"}},
                "required": ["id"],
            },
        ),
        types.Tool(
            name="update_database",
            description="Update a database",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"},
                    "name": {"type": "string", "description": "Database name"},
                    "description": {
                        "type": "string",
                        "description": "Database description",
                    },
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="delete_database",
            description="Delete a database",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Database ID"}},
                "required": ["id"],
            },
        ),
        # Worker Set tools
        types.Tool(
            name="list_worker_sets",
            description="List all worker sets",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="create_worker_set",
            description="Create a new worker set",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Worker set name"},
                    "worker_count": {
                        "type": "integer",
                        "description": "Number of workers",
                    },
                },
                "required": ["name", "worker_count"],
            },
        ),
        types.Tool(
            name="get_worker_set",
            description="Get a specific worker set by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="delete_worker_set",
            description="Delete a worker set",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="get_worker_set_actions",
            description="List the actions for a worker-set",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"],
            },
        ),
        # Worker Group tools
        types.Tool(
            name="list_worker_groups",
            description="List all worker groups",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # Worker tools
        types.Tool(
            name="list_available_actions",
            description="List actions (or workers) from all worker-sets",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # Workflow tools
        types.Tool(
            name="list_workflows",
            description="List all workflows",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="create_workflow",
            description="Create and run a new workflow",
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
                                    "description": "Alias for this job (used for referencing in inputs)",
                                },
                                "config": {
                                    "type": "object",
                                    "description": "Job configuration (can contain any action config)",
                                },
                                "input": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of input aliases this job depends on",
                                },
                                "replicas": {
                                    "type": "integer",
                                    "default": 1,
                                    "description": "Number of replicas for this job",
                                },
                                "worker_name": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                    "description": "Name of the worker to execute this job",
                                },
                                "worker_version": {
                                    "type": "string",
                                    "description": "Version of the worker",
                                },
                            },
                            "required": [
                                "alias",
                                "config",
                                "input",
                                "worker_name",
                                "worker_version",
                            ],
                        },
                        "minItems": 1,
                        "description": "Array of jobs that make up the workflow",
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                "description": "Workflow name",
                            },
                            "labels": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN _-]{1,64}$",
                                },
                                "description": "Key-value labels for the workflow",
                            },
                        },
                        "required": ["name", "labels"],
                        "description": "Workflow metadata",
                    },
                },
                "required": ["jobs", "metadata"],
                "description": "Workflow definition with jobs and metadata",
            },
        ),
        types.Tool(
            name="get_workflow",
            description="Get a specific workflow by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Workflow ID"}},
                "required": ["id"],
            },
        ),
        types.Tool(
            name="update_workflow",
            description="Update a workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID"},
                    "name": {"type": "string", "description": "Workflow name"},
                    "definition": {
                        "type": "object",
                        "description": "Workflow definition",
                    },
                },
                "required": ["id"],
            },
        ),
        # Models tools
        types.Tool(
            name="list_models",
            description="List all models",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="upload_model",
            description="Upload a new model",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name"},
                    "metadata": {"type": "object", "description": "Model metadata"},
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_model",
            description="Get a specific model by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Model ID"}},
                "required": ["id"],
            },
        ),
        types.Tool(
            name="delete_model",
            description="Delete a model",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Model ID"}},
                "required": ["id"],
            },
        ),
        # Organization tools
        types.Tool(
            name="get_active_organization",
            description="Get active organization",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="list_organizations",
            description="List all organizations",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # Project tools
        types.Tool(
            name="get_active_project",
            description="Get active project",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="list_projects",
            description="List all projects",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="create_project",
            description="Create a new project",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {
                        "type": "string",
                        "description": "Project description",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_project",
            description="Get a specific project by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Project ID"}},
                "required": ["id"],
            },
        ),
        types.Tool(
            name="update_project",
            description="Update a project",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Project ID"},
                    "name": {"type": "string", "description": "Project name"},
                    "description": {
                        "type": "string",
                        "description": "Project description",
                    },
                },
                "required": ["id"],
            },
        ),
        # Dataset tools
        types.Tool(
            name="list_datasets",
            description="List all datasets",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="create_dataset",
            description="Create a new dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {
                        "type": "string",
                        "description": "Dataset description",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_dataset",
            description="Get a specific dataset by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Dataset ID"}},
                "required": ["id"],
            },
        ),
        types.Tool(
            name="update_dataset",
            description="Update a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"},
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {
                        "type": "string",
                        "description": "Dataset description",
                    },
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="delete_dataset",
            description="Delete a dataset",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Dataset ID"}},
                "required": ["id"],
            },
        ),
        # Dataset schema tools
        types.Tool(
            name="get_dataset_schema",
            description="Get the schema for a dataset",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Dataset ID"}},
                "required": ["id"],
            },
        ),
        # Query tools
        types.Tool(
            name="execute_query",
            description="Execute a query and return results in CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": 'The SQL query to execute. REQUIRED SQL SYNTAX: Use dataset ID wrapped in double quotes as table name (e.g., "rWBroVVFla7I0Fv9nBZXh") and wrap column names in backticks (e.g., `column_name`). Example: SELECT `timestamp`, `value` FROM "rWBroVVFla7I0Fv9nBZXh" WHERE `status` = \'active\'',
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project ID to execute the query in",
                    },
                },
                "required": ["query"],
            },
        ),
    ]

    # Add Manta tools only if Manta client is initialized
    if manta_client:
        manta_tools = [
            # Manta Service - Data Manipulation tools (Incident Injection v2)
            types.Tool(
                name="create_incident_dataset",
                description="""Create a new dataset with injected incidents. There are 4 suppported incidents and each incident type requires a specific configuration schema:

INCIDENT TYPE TO CONFIG MAPPING:
1. 'instantaneous-spike-data' → InstantaneousSpikeIncidentConfig
   - Injects sudden spikes at a single timestamp
   - Required fields: impacted_metadata_predicate, impacted_measurement, absolute_magnitude, timestamp_column, timestamp

2. 'sustained-magnitude-change-data' → SustainedMagnitudeChangeIncidentConfig
   - Applies a sustained change over a time period
   - Required fields: impacted_metadata_predicate, impacted_measurement, delta_magnitude, timestamp_column, start_timestamp, end_timestamp

3. 'data-outage-data' → DataOutageIncidentConfig
   - Creates data gaps/outages over a time period
   - Required fields: impacted_metadata_predicate, impacted_measurement, absolute_magnitude, timestamp_column, start_timestamp, end_timestamp

4. 'value-ramp-data' → ValueRampIncidentConfig
   - Applies gradual ramping changes over a time period
   - Required fields: impacted_metadata_predicate, impacted_measurement, end_absolute_magnitude, slope, timestamp_column, start_timestamp, end_timestamp

IMPORTANT: The incident_config object schema MUST match the selected incident_type.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Source time-series dataset ID",
                        },
                        "incident_type": {
                            "type": "string",
                            "enum": [
                                "instantaneous-spike-data",
                                "sustained-magnitude-change-data",
                                "data-outage-data",
                                "value-ramp-data",
                            ],
                            "description": "Type of incident to inject. MUST match the incident_config schema: instantaneous-spike-data→InstantaneousSpikeIncidentConfig, sustained-magnitude-change-data→SustainedMagnitudeChangeIncidentConfig, data-outage-data→DataOutageIncidentConfig, value-ramp-data→ValueRampIncidentConfig",
                        },
                        "incident_config": {
                            "type": "object",
                            "description": "Configuration object that MUST match the incident_type. See tool description for the exact mapping of incident_type to config schema.",
                            "oneOf": [
                                {
                                    "title": "InstantaneousSpikeIncidentConfig (for incident_type='instantaneous-spike-data')",
                                    "type": "object",
                                    "properties": {
                                        "impacted_metadata_predicate": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "column_name": {"type": "string"},
                                                    "value": {
                                                        "description": "Any value type"
                                                    },
                                                },
                                                "required": ["column_name", "value"],
                                            },
                                            "description": "List of metadata predicates to identify impacted rows",
                                        },
                                        "impacted_measurement": {
                                            "type": "string",
                                            "description": "Name of the measurement column to inject spike into",
                                        },
                                        "absolute_magnitude": {
                                            "type": ["number", "integer"],
                                            "description": "Absolute spike value to inject",
                                        },
                                        "timestamp_column": {
                                            "type": "string",
                                            "description": "Name of the timestamp column",
                                        },
                                        "timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "Timestamp when spike occurs (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                    },
                                    "required": [
                                        "impacted_metadata_predicate",
                                        "impacted_measurement",
                                        "absolute_magnitude",
                                        "timestamp_column",
                                        "timestamp",
                                    ],
                                },
                                {
                                    "title": "SustainedMagnitudeChangeIncidentConfig (for incident_type='sustained-magnitude-change-data')",
                                    "type": "object",
                                    "properties": {
                                        "impacted_metadata_predicate": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "column_name": {"type": "string"},
                                                    "value": {
                                                        "description": "Any value type"
                                                    },
                                                },
                                                "required": ["column_name", "value"],
                                            },
                                            "description": "List of metadata predicates to identify impacted rows",
                                        },
                                        "impacted_measurement": {
                                            "type": "string",
                                            "description": "Name of the measurement column to modify",
                                        },
                                        "delta_magnitude": {
                                            "type": ["number", "integer"],
                                            "description": "Delta value to add to measurements during incident period",
                                        },
                                        "timestamp_column": {
                                            "type": "string",
                                            "description": "Name of the timestamp column",
                                        },
                                        "start_timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "Start timestamp of sustained change (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                        "end_timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "End timestamp of sustained change (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                    },
                                    "required": [
                                        "impacted_metadata_predicate",
                                        "impacted_measurement",
                                        "delta_magnitude",
                                        "timestamp_column",
                                        "start_timestamp",
                                        "end_timestamp",
                                    ],
                                },
                                {
                                    "title": "DataOutageIncidentConfig (for incident_type='data-outage-data')",
                                    "type": "object",
                                    "properties": {
                                        "impacted_metadata_predicate": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "column_name": {"type": "string"},
                                                    "value": {
                                                        "description": "Any value type"
                                                    },
                                                },
                                                "required": ["column_name", "value"],
                                            },
                                            "description": "List of metadata predicates to identify impacted rows",
                                        },
                                        "impacted_measurement": {
                                            "type": "string",
                                            "description": "Name of the measurement column to set to outage value",
                                        },
                                        "absolute_magnitude": {
                                            "type": ["number", "integer"],
                                            "description": "Value to set during outage (typically 0 or null-equivalent)",
                                        },
                                        "timestamp_column": {
                                            "type": "string",
                                            "description": "Name of the timestamp column",
                                        },
                                        "start_timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "Start timestamp of outage (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                        "end_timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "End timestamp of outage (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                    },
                                    "required": [
                                        "impacted_metadata_predicate",
                                        "impacted_measurement",
                                        "absolute_magnitude",
                                        "timestamp_column",
                                        "start_timestamp",
                                        "end_timestamp",
                                    ],
                                },
                                {
                                    "title": "ValueRampIncidentConfig (for incident_type='value-ramp-data')",
                                    "type": "object",
                                    "properties": {
                                        "impacted_metadata_predicate": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "column_name": {"type": "string"},
                                                    "value": {
                                                        "description": "Any value type"
                                                    },
                                                },
                                                "required": ["column_name", "value"],
                                            },
                                            "description": "List of metadata predicates to identify impacted rows",
                                        },
                                        "impacted_measurement": {
                                            "type": "string",
                                            "description": "Name of the measurement column to ramp",
                                        },
                                        "end_absolute_magnitude": {
                                            "type": ["number", "integer"],
                                            "description": "Final absolute value at end of ramp",
                                        },
                                        "slope": {
                                            "type": ["number", "integer"],
                                            "description": "Rate of change per time unit",
                                        },
                                        "timestamp_column": {
                                            "type": "string",
                                            "description": "Name of the timestamp column",
                                        },
                                        "start_timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "Start timestamp of ramp (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                        "end_timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "End timestamp of ramp (ISO 8601 format, e.g., '2024-01-15T10:30:00Z' or '2024-01-15T10:30:00-05:00')",
                                        },
                                    },
                                    "required": [
                                        "impacted_metadata_predicate",
                                        "impacted_measurement",
                                        "end_absolute_magnitude",
                                        "slope",
                                        "timestamp_column",
                                        "start_timestamp",
                                        "end_timestamp",
                                    ],
                                },
                            ],
                        },
                        "organization_id": {
                            "type": "string",
                            "description": "Organization ID (required by Manta API)",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (required by Manta API)",
                        },
                    },
                    "required": [
                        "dataset_id",
                        "incident_type",
                        "incident_config",
                        "organization_id",
                        "project_id",
                    ],
                },
            ),
            types.Tool(
                name="generate_incident_prompts",
                description="Generate analysis prompts for an incident dataset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Incident dataset ID (from create_incident_dataset)",
                        },
                        "organization_id": {
                            "type": "string",
                            "description": "Organization ID (required by Manta API)",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (required by Manta API)",
                        },
                    },
                    "required": ["dataset_id", "organization_id", "project_id"],
                },
            ),
            types.Tool(
                name="list_incident_datasets",
                description="List all incident datasets derived from a source dataset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Source dataset ID",
                        },
                        "organization_id": {
                            "type": "string",
                            "description": "Organization ID (required by Manta API)",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (required by Manta API)",
                        },
                    },
                    "required": ["dataset_id", "organization_id", "project_id"],
                },
            ),
            types.Tool(
                name="get_incident_prompts",
                description="Retrieve previously generated prompts for an incident dataset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Incident dataset ID",
                        },
                        "organization_id": {
                            "type": "string",
                            "description": "Organization ID (required by Manta API)",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (required by Manta API)",
                        },
                    },
                    "required": ["dataset_id", "organization_id", "project_id"],
                },
            ),
        ]
        tools.extend(manta_tools)

    # Add SDK-specific tools
    tools.extend(
        [
            types.Tool(
                name="obtain_train_config",
                description="Generate training configuration for synthetic data models. Returns a cached train_config_id for use in start_training_workflow. Supports TabGAN (rf_tab_gan) and TimeGAN (rf_time_gan).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Dataset ID to generate config for",
                        },
                        "model_type": {
                            "type": "string",
                            "enum": ["rf_tab_gan", "rf_time_gan"],
                            "description": "Model type: 'rf_tab_gan' for tabular data or 'rf_time_gan' for time-series data",
                        },
                    },
                    "required": ["dataset_id", "model_type"],
                },
            ),
            types.Tool(
                name="start_training_workflow",
                description="Build and start a training workflow using a cached config. Automatically detects model type from train_config structure. Use this after obtaining train_config_id from obtain_train_config.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Dataset ID to train on",
                        },
                        "train_config_id": {
                            "type": "string",
                            "description": "Config ID from obtain_train_config (retrieves cached config)",
                        },
                    },
                    "required": ["dataset_id", "train_config_id"],
                },
            ),
            types.Tool(
                name="update_train_config",
                description="Update training configuration. Modify model hyperparameters (epochs, batch_size, etc.) and/or encoder field classifications (categorical, continuous, ignore). See https://docs.rockfish.ai/model-train.html for config structure.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "train_config_id": {
                            "type": "string",
                            "description": "Config ID from obtain_train_config",
                        },
                        "updates": {
                            "type": "object",
                            "properties": {
                                "model_config": {
                                    "type": "object",
                                    "description": "Model hyperparameters to update (e.g., epochs, batch_size, g_lr, d_lr)",
                                    "additionalProperties": True,
                                },
                                "encoder_config": {
                                    "type": "object",
                                    "properties": {
                                        "metadata": {
                                            "type": "object",
                                            "description": "Field name to type mappings. Type: 'categorical', 'continuous', 'ignore', or 'session' (TimeGAN only)",
                                            "additionalProperties": {"type": "string"},
                                        },
                                        "measurements": {
                                            "type": "object",
                                            "description": "(TimeGAN only, TODO: not yet implemented) Time-series measurement field classifications",
                                            "additionalProperties": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            "description": "Updates to apply: model_config for hyperparameters, encoder_config for field types",
                        },
                    },
                    "required": ["train_config_id", "updates"],
                },
            ),
            types.Tool(
                name="get_workflow_logs",
                description="Stream logs from a workflow execution with configurable log level and collection timeout. Collects logs for specified duration (default 10s), useful for monitoring workflow progress and debugging. Call repeatedly to continue collecting logs from running workflows.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "Workflow ID to get logs from",
                        },
                        "log_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARN", "ERROR"],
                            "description": "Log level filter. Default: INFO",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Collection duration in seconds. Default: 10",
                        },
                    },
                    "required": ["workflow_id"],
                },
            ),
            types.Tool(
                name="get_trained_model_id",
                description="Extract the trained model ID from a completed training workflow. Only works on COMPLETED or FINALIZED workflows.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "Training workflow ID",
                        }
                    },
                    "required": ["workflow_id"],
                },
            ),
            types.Tool(
                name="start_generation_workflow",
                description="Start a workflow to generate synthetic data from a trained TabGAN model.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Trained model ID from get_trained_model_id",
                        }
                    },
                    "required": ["model_id"],
                },
            ),
            types.Tool(
                name="obtain_synthetic_dataset_id",
                description="Extract the synthetic dataset ID from a completed generation workflow. Only works on COMPLETED or FINALIZED workflows.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "generation_workflow_id": {
                            "type": "string",
                            "description": "Generation workflow ID from start_generation_workflow",
                        }
                    },
                    "required": ["generation_workflow_id"],
                },
            ),
            types.Tool(
                name="plot_distribution",
                description="Generate distribution plot comparing datasets for a specific column. Returns base64-encoded PNG image. Automatically chooses bar plot (categorical) or KDE plot (numerical).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of dataset IDs to compare (e.g., [real_dataset_id, synthetic_dataset_id])",
                        },
                        "column_name": {
                            "type": "string",
                            "description": "Column name to plot distribution for",
                        },
                    },
                    "required": ["dataset_ids", "column_name"],
                },
            ),
            types.Tool(
                name="get_marginal_distribution_score",
                description="Calculate marginal distribution score between two datasets (real vs synthetic). Lower score means more similar distributions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Exactly 2 dataset IDs: [real_dataset_id, synthetic_dataset_id]",
                        }
                    },
                    "required": ["dataset_ids"],
                },
            ),
        ]
    )

    return tools


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool calls to Rockfish API, SDK, and Manta service."""

    # Route SDK-specific tools to sdk_client
    sdk_tools = [
        "obtain_train_config",
        "update_train_config",
        "start_training_workflow",
        "get_workflow_logs",
        "get_trained_model_id",
        "start_generation_workflow",
        "obtain_synthetic_dataset_id",
        "plot_distribution",
        "get_marginal_distribution_score",
    ]

    if name in sdk_tools:
        if not sdk_client:
            return [
                types.TextContent(
                    type="text",
                    text="SDK client not initialized. Please check your API credentials.",
                )
            ]

        try:
            result = await sdk_client.call_endpoint(name, arguments)

            # Handle image responses (e.g., from plot_distribution)
            if isinstance(result, dict) and "image" in result and "mimeType" in result:
                # Return both image and descriptive text
                response_parts = [
                    types.ImageContent(
                        type="image", data=result["image"], mimeType=result["mimeType"]
                    )
                ]

                # Add descriptive text if available
                if "column_name" in result:
                    dataset_count = len(result.get("dataset_ids", []))
                    description = (
                        f"Distribution plot for column '{result['column_name']}'"
                    )
                    if dataset_count > 0:
                        description += f" comparing {dataset_count} dataset(s)"
                    response_parts.append(
                        types.TextContent(type="text", text=description)
                    )

                return response_parts

            # Default: return as text
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Error calling {name}: {e}")
            return [
                types.TextContent(type="text", text=f"Error calling {name}: {str(e)}")
            ]

    # Route Manta tools to manta_client (includes v2 tools without manta_ prefix)
    manta_v2_tools = [
        "create_incident_dataset",
        "generate_incident_prompts",
        "list_incident_datasets",
        "get_incident_prompts",
    ]
    if name in manta_v2_tools:
        if not manta_client:
            return [
                types.TextContent(
                    type="text",
                    text="Manta client not initialized. Please check your API credentials.",
                )
            ]

        try:
            result = await manta_client.call_endpoint(name, arguments)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Error calling {name}: {e}")
            return [
                types.TextContent(type="text", text=f"Error calling {name}: {str(e)}")
            ]

    # Route all other tools to rockfish_client
    if not rockfish_client:
        return [
            types.TextContent(
                type="text",
                text="Rockfish client not initialized. Please check your API credentials.",
            )
        ]

    try:
        result = await rockfish_client.call_endpoint(name, arguments)
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        logger.error(f"Error calling {name}: {e}")
        return [types.TextContent(type="text", text=f"Error calling {name}: {str(e)}")]


async def main():
    global rockfish_client, manta_client, sdk_client

    # Initialize Rockfish client
    api_key = os.getenv("ROCKFISH_API_KEY")
    # Support both new API_URL and legacy BASE_URL variable names for backwards compatibility
    api_url = os.getenv("ROCKFISH_API_URL") or os.getenv(
        "ROCKFISH_BASE_URL", "https://api.rockfish.ai"
    )
    organization_id = os.getenv("ROCKFISH_ORGANIZATION_ID", None)
    project_id = os.getenv("ROCKFISH_PROJECT_ID", None)

    if not api_key:
        logger.error("ROCKFISH_API_KEY environment variable is required")
        return

    rockfish_client = RockfishClient(
        api_key=api_key,
        api_url=api_url,
        organization_id=organization_id,
        project_id=project_id,
    )

    # Initialize SDK client (always initialized when API key is present)
    sdk_client = RockfishSDKClient(
        API_KEY=api_key,
        API_URL=api_url,
        ORGANIZATION_ID=organization_id,
        PROJECT_ID=project_id,
    )
    logger.info("SDK client initialized")

    # Initialize Manta client only if MANTA_API_URL is configured
    # Support both new API_URL and legacy BASE_URL variable names for backwards compatibility
    manta_api_url = os.getenv("MANTA_API_URL") or os.getenv("MANTA_BASE_URL")
    if manta_api_url:
        manta_client = MantaClient(api_key=api_key, api_url=manta_api_url)
        logger.info(f"Manta client initialized with API URL: {manta_api_url}")
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

    # Cleanup connections
    if sdk_client:
        await sdk_client.close()


def cli():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
