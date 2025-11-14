import logging
from typing import Any, Dict
import io
import base64

# Set matplotlib to non-interactive backend BEFORE importing pyplot
# This prevents GUI windows and Python icon from appearing in dock
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rockfish as rf
import rockfish.labs
import rockfish.labs.vis
import rockfish.actions as ra
from dotenv import load_dotenv
from .sdk_docstring_extractor import get_extractor
import pyarrow as pa

logger = logging.getLogger(__name__)
load_dotenv()

class RockfishSDKClient:
    """SDK client for interacting with Rockfish using the official Python SDK.

    This client uses the official Rockfish SDK (rockfish package) for operations
    that are well-supported, providing better type safety and future-proofing.

    Uses Connection.from_env() to automatically read configuration from environment variables:
    - ROCKFISH_API_KEY
    - ROCKFISH_API_URL (optional, defaults to https://api.rockfish.ai)
    - ROCKFISH_ORGANIZATION_ID (optional)
    - ROCKFISH_PROJECT_ID (optional)
    """

    def __init__(self):
        """Initialize SDK client using environment variables via Connection.from_env()."""
        self._conn = None

    async def _get_connection(self):
        """Get or create a connection instance."""
        if self._conn is None:
            self._conn = rf.Connection.from_env() # TODO: use the same way to get api key, api url as HTTP client?
            logger.info("SDK connection initialized from environment")
        return self._conn

    async def close(self):
        """Close the SDK connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def call_endpoint(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool calls to appropriate SDK methods.

        Raises:
            NotImplementedError: If the operation is not supported by the SDK
            ValueError: If unknown tool name
        """
        conn = await self._get_connection()

        # Organization endpoints
        if tool_name == "get_active_organization":
            org = await conn.active_organization()
            return {"id": org.id, "name": org.name}

        elif tool_name == "list_organizations":
            orgs = []
            async for org in conn.organizations():
                orgs.append({"id": org.id, "name": org.name})
            return {"organizations": orgs}

        # Project endpoints
        elif tool_name == "get_active_project":
            project = await conn.active_project()
            return {
                "id": project.id,
                "name": project.name,
                "default": project.default,
                "created_at": project.create_time.isoformat() if project.create_time else None,
            }

        elif tool_name == "list_projects":
            projects = []
            async for project in conn.projects():
                projects.append({
                    "id": project.id,
                    "name": project.name,
                    "default": project.default,
                    "created_at": project.create_time.isoformat() if project.create_time else None,
                })
            return {"projects": projects}

        elif tool_name == "create_project":
            name = arguments["name"]
            project = await conn.create_project(name)
            return {
                "id": project.id,
                "name": project.name,
                "default": project.default,
                "created_at": project.create_time.isoformat() if project.create_time else None,
            }

        elif tool_name == "get_project":
            # SDK doesn't have direct get_project by ID, need to list and filter
            project_id = arguments["id"]
            async for project in conn.projects():
                if project.id == project_id:
                    return {
                        "id": project.id,
                        "name": project.name,
                        "default": project.default,
                        "created_at": project.create_time.isoformat() if project.create_time else None,
                    }
            raise ValueError(f"Project not found: {project_id}")

        # Workflow endpoints
        elif tool_name == "list_workflows":
            workflows = []
            limit = arguments.get("limit", 10)
            async for workflow in conn.workflows(limit=limit):
                workflows.append({
                    "id": workflow.id(),
                    "status": await workflow.status(),
                })
            return {"workflows": workflows}

        elif tool_name == "create_workflow":
            # Complex operation - requires WorkflowItem objects
            # For now, pass through arguments to SDK
            items = arguments.get("items", [])
            worker_group = arguments.get("worker_group")
            name = arguments.get("name")
            labels = arguments.get("labels")

            workflow = await conn.create_workflow(
                items=items,
                worker_group=worker_group,
                name=name,
                labels=labels
            )
            return {"id": workflow.id(), "status": await workflow.status()}

        elif tool_name == "get_workflow":
            workflow_id = arguments["id"]
            workflow = await conn.get_workflow(workflow_id)
            return {"id": workflow.id(), "status": await workflow.status()}

        elif tool_name == "visualize_workflow":
            workflow_id = arguments["id"]
            direction = arguments.get("direction", "LR")
            workflow = await conn.get_workflow(workflow_id)

            # Call the SDK's mermaid() method to generate diagram
            mermaid_diagram = workflow.mermaid(direction=direction)

            # Return the Mermaid markdown string
            return {
                "workflow_id": workflow.id(),
                "mermaid": str(mermaid_diagram)
            }
        # not sure if it is visuable as no id
        elif tool_name == "visualize_workflow_builder":
            actions_config = arguments["actions"]
            direction = arguments.get("direction", "LR")

            # Create WorkflowBuilder
            builder = rf.WorkflowBuilder()

            # Dynamically instantiate action classes from config
            action_instances = []
            for action_def in actions_config:
                action_class_name = action_def["action_class"]
                config = action_def.get("config", {})

                # Get action class from rockfish.actions module
                if not hasattr(ra, action_class_name):
                    raise ValueError(f"Unknown action class: {action_class_name}")

                action_class = getattr(ra, action_class_name)
                action_instance = action_class(**config)
                action_instances.append(action_instance)

            # Connect actions sequentially using add_path
            if action_instances:
                builder.add_path(*action_instances)

            # Generate visualization WITHOUT executing
            mermaid_diagram = builder.mermaid(direction=direction)

            return {
                "mermaid": str(mermaid_diagram),
                "action_count": len(action_instances)
            }

        elif tool_name == "plot_dataset_distribution":
            def _fig_to_base64(fig):
                """Convert a figure(plot) to a base64 string"""
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
                img_str = base64.b64encode(buf.getbuffer()).decode("utf-8")
                buf.close()
                plt.close(fig.fig)  # Close the underlying matplotlib figure to free memory
                return img_str

            dataset_id = arguments["dataset_id"]
            column_name = arguments["column_name"]
            bins = arguments.get("bins", 30)

            # Load dataset and convert to LocalDataset
            dataset = await conn.get_dataset(dataset_id)
            local_dataset = await dataset.to_local(conn)

            # Workaround for SDK bug in plot_distribution():
            # The SDK tries to access chunked_array.num_rows which doesn't exist
            # We implement the same logic but use table.num_rows instead
            table = local_dataset.table
            field_type = table[column_name].type

            # Choose plot type based on data characteristics (fixing SDK bug)
            if pa.types.is_string(field_type):
                # Categorical/string data → bar plot
                fig = rockfish.labs.vis.plot_bar([local_dataset], column_name)
            elif table.num_rows <= 10:  # Fix: Use table.num_rows, not chunked_array.num_rows
                # Very few rows → bar plot
                fig = rockfish.labs.vis.plot_bar([local_dataset], column_name)
            else:
                # Numerical data with enough rows → KDE plot
                fig = rockfish.labs.vis.plot_kde([local_dataset], column_name)

            img_base64 = _fig_to_base64(fig)

            # Return image data
            return {
                "image": img_base64,
                "mimeType": "image/png",
                "dataset_id": dataset_id,
                "column_name": column_name,
                "bins": bins
            }

        elif tool_name == "update_workflow":
            workflow_id = arguments.pop("id")
            await conn.update_workflow(workflow_id, **arguments)
            return {"success": True}

        # Dataset endpoints
        elif tool_name == "list_datasets":
            datasets = []
            limit = arguments.get("limit", 10)
            async for dataset in conn.datasets(limit=limit):
                datasets.append({
                    "id": dataset.id,
                    "url": str(dataset.url),
                    **dataset.metadata
                })
            return {"datasets": datasets}

        elif tool_name == "get_dataset":
            dataset_id = arguments["id"]
            dataset = await conn.get_dataset(dataset_id)
            return {
                "id": dataset.id,
                "url": str(dataset.url),
                **dataset.metadata
            }

        elif tool_name == "delete_dataset":
            dataset_id = arguments["id"]
            await conn.delete_dataset(dataset_id)
            return {"success": True}

        # Query endpoints
        elif tool_name == "execute_query":
            query = arguments["query"]
            result = await conn.query_datasets(query)
            # Convert LocalDataset to string representation
            return {"result": str(result)}

        elif tool_name == "query_dataset":
            dataset_id = arguments["id"]
            query = arguments.get("query")
            result = await conn.query_dataset(dataset_id, query)
            # Convert PyArrow table to string representation
            return {"result": str(result)}

        # Model endpoints
        elif tool_name == "list_models":
            models = []
            limit = arguments.get("limit", 10)
            async for model in conn.models(limit=limit):
                models.append({
                    "id": model.id,
                    "labels": model.labels,
                    "created_at": model.create_time.isoformat() if model.create_time else None,
                    "size_bytes": model.size_bytes,
                })
            return {"models": models}

        elif tool_name == "get_model":
            model_id = arguments["id"]
            model = await conn.get_model(model_id)
            return {
                "id": model.id,
                "labels": model.labels,
                "created_at": model.create_time.isoformat() if model.create_time else None,
                "size_bytes": model.size_bytes,
            }

        # Tabular dataset properties extraction (SDK workflow-based)
        elif tool_name == "extract_tabular_properties":
            dataset_id = arguments["dataset_id"]
            detect_pii = arguments.get("detect_pii", False)
            detect_association_rules = arguments.get("detect_association_rules", False)
            association_threshold = arguments.get("association_threshold", 0.95)

            # Build workflow: DatasetLoad → TabPropertyExtractor → DatasetSave
            builder = rf.WorkflowBuilder()
            dataset_load = ra.DatasetLoad(dataset_id=dataset_id)
            extract_tab_props = ra.TabPropertyExtractor(
                detect_pii=detect_pii,
                detect_association_rules=detect_association_rules,
                association_threshold=association_threshold,
            )
            dataset_save = ra.DatasetSave(name=f"dataset_{dataset_id}_with_props")

            builder.add_path(dataset_load, extract_tab_props, dataset_save)
            workflow = await builder.start(conn)
            logger.info(f"Started property extraction workflow: {workflow.id()}")

            # Wait for workflow completion
            await workflow.wait(raise_on_failure=True)
            logger.info(f"Workflow {workflow.id()} completed successfully")

            # Extract properties from resulting dataset
            # TODO: maybe workflow.datasets().first()?
            dataset = None
            async for ds in workflow.datasets():
                dataset_with_props_id = ds.id
                dataset = await ds.to_local(conn)
            dataset_properties = dataset.table_metadata().dataset_properties

            # Extract field properties for all fields
            field_properties_map = {}
            for field in dataset.table.schema:
                field_name = field.name
                field_properties = dataset.get_field_properties(field_name)
                field_properties_map[field_name] = field_properties

            # Return unstructured data (convert to dicts)
            return {
                "dataset_properties": rf.converter.unstructure(dataset_properties),
                "field_properties_map": rf.converter.unstructure(field_properties_map),
                "workflow_id": workflow.id(),
                "output_dataset_id": dataset_with_props_id
            }

        # Timeseries dataset properties extraction (SDK workflow-based)
        elif tool_name == "extract_timeseries_properties":
            dataset_id = arguments["dataset_id"]
            timestamp = arguments["timestamp"]  # Required for timeseries
            session_fields = arguments.get("session_fields", [])
            metadata_fields = arguments.get("metadata_fields", None)
            detect_metadata_fields = arguments.get("detect_metadata_fields", False)
            detect_pii = arguments.get("detect_pii", False)
            detect_association_rules = arguments.get("detect_association_rules", False)
            association_threshold = arguments.get("association_threshold", 0.95)

            # Validate mutual exclusivity
            if metadata_fields is not None and detect_metadata_fields:
                raise ValueError(
                    "Cannot specify both 'metadata_fields' and 'detect_metadata_fields=True'. "
                    "Use one or the other."
                )

            # Build workflow: DatasetLoad → TimePropertyExtractor → DatasetSave
            builder = rf.WorkflowBuilder()
            dataset_load = ra.DatasetLoad(dataset_id=dataset_id)
            extract_time_props = ra.TimePropertyExtractor(
                timestamp=timestamp,
                session_fields=session_fields,
                metadata_fields=metadata_fields,
                detect_metadata_fields=detect_metadata_fields,
                detect_pii=detect_pii,
                detect_association_rules=detect_association_rules,
                association_threshold=association_threshold,
            )
            dataset_save = ra.DatasetSave(name=f"dataset_{dataset_id}_with_props")

            builder.add_path(dataset_load, extract_time_props, dataset_save)
            workflow = await builder.start(conn)
            logger.info(f"Started property extraction workflow: {workflow.id()}")

            # Wait for workflow completion
            await workflow.wait(raise_on_failure=True)
            logger.info(f"Workflow {workflow.id()} completed successfully")

            # Extract properties from resulting dataset
            dataset = None
            async for ds in workflow.datasets():
                dataset_with_props_id = ds.id
                dataset = await ds.to_local(conn)
            dataset_properties = dataset.table_metadata().dataset_properties

            # Extract field properties for all fields
            field_properties_map = {}
            for field in dataset.table.schema:
                field_name = field.name
                field_properties = dataset.get_field_properties(field_name)
                field_properties_map[field_name] = field_properties

            # Return unstructured data (convert to dicts)
            return {
                "dataset_properties": rf.converter.unstructure(dataset_properties),
                "field_properties_map": rf.converter.unstructure(field_properties_map),
                "workflow_id": workflow.id(),
                "output_dataset_id": dataset_with_props_id
            }

        # SDK Documentation tools
        elif tool_name == "get_sdk_docs":
            extractor = get_extractor()
            method_name = arguments.get("method_name")
            if method_name:
                result = extractor.format_connection_docs_markdown(method_name)
            else:
                result = extractor.format_connection_docs_markdown()
            return {"documentation": result}

        elif tool_name == "list_sdk_actions":
            extractor = get_extractor()
            result = extractor.format_action_list_markdown()
            return {"actions": result}

        elif tool_name == "get_action_docs":
            extractor = get_extractor()
            action_name = arguments["action_name"]
            result = extractor.format_action_doc_markdown(action_name)
            return {"documentation": result}

        elif tool_name == "get_action_config_schema":
            extractor = get_extractor()
            action_name = arguments["action_name"]
            schema = extractor.get_action_schema(action_name)
            if schema:
                return {
                    "action": action_name,
                    "config_schema": schema,
                    "summary": extractor.get_action_summary(action_name)
                }
            else:
                available_actions = ', '.join(list(extractor.get_all_action_schemas().keys())[:20])
                return {
                    "error": f"Action '{action_name}' not found or has no config schema.",
                    "available_actions": available_actions + "..."
                }

        # Operations not supported by SDK - should use HTTP client instead
        elif tool_name in [
            "update_project", "create_dataset", "update_dataset", "get_dataset_schema",
            "upload_model", "delete_model"
        ]:
            raise NotImplementedError(
                f"{tool_name} not supported by SDK, use HTTP client instead"
            )

        else:
            raise ValueError(f"Unknown tool: {tool_name}")
