import logging
from typing import Any, Dict
import rockfish as rf
import rockfish.labs
import rockfish.actions as ra
from dotenv import load_dotenv

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
        elif tool_name == "extract_tabular_properties_sdk":
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
