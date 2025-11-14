"""SDK Docstring Extractor for exposing Rockfish SDK documentation via MCP.

This module extracts docstrings from the Rockfish Python SDK and formats them
for display in MCP tool descriptions and dedicated documentation tools.
"""

import inspect
from typing import Dict, List, Tuple, Optional
import rockfish as rf
import rockfish.actions as ra


class SDKDocstringExtractor:
    """Extract and cache SDK docstrings and schemas for MCP exposure.

    This class extracts:
    - Docstrings from rockfish.AbstractConnection methods
    - Docstrings from rockfish.actions classes (all 63+ action classes)
    - Config schemas from action classes (JSON schema format)

    All data is cached on initialization for fast access.
    """

    def __init__(self):
        """Initialize the extractor and cache all SDK docstrings and schemas."""
        self._connection_methods: Dict[str, str] = {}
        self._action_classes: Dict[str, str] = {}
        self._action_summaries: Dict[str, str] = {}
        self._action_schemas: Dict[str, dict] = {}
        self._extract_all()

    def _extract_all(self):
        """Extract all SDK docstrings and schemas during initialization."""
        self._extract_connection_methods()
        self._extract_action_classes()
        self._extract_action_schemas()

    @staticmethod
    def _schema_to_dict(schema_obj) -> dict:
        """Convert a Rockfish schema object to a dictionary.

        Args:
            schema_obj: A Rockfish schema object (from action_class.config_schema())

        Returns:
            Dictionary representation of the schema
        """
        if schema_obj is None:
            return None

        # Handle NoDefaultType
        if hasattr(schema_obj, '__class__') and schema_obj.__class__.__name__ == 'NoDefaultType':
            return None  # No default

        # If it's already a basic type, return it
        if isinstance(schema_obj, (str, int, float, bool, list)):
            return schema_obj

        # Handle complex objects that aren't schema objects (like ModelConfig default values)
        # Just return their string representation
        if not hasattr(schema_obj, 'type'):
            return str(schema_obj)

        result = {}

        # Get the type
        if hasattr(schema_obj, 'type'):
            result['type'] = schema_obj.type

        # Get required fields
        if hasattr(schema_obj, 'required') and schema_obj.required:
            result['required'] = schema_obj.required

        # Get properties (recursively convert)
        if hasattr(schema_obj, 'properties') and schema_obj.properties:
            result['properties'] = {}
            for prop_name, prop_schema in schema_obj.properties.items():
                result['properties'][prop_name] = SDKDocstringExtractor._schema_to_dict(prop_schema)

        # Get default value
        if hasattr(schema_obj, 'default'):
            default_val = schema_obj.default
            if default_val is not None and not (hasattr(default_val, '__class__') and default_val.__class__.__name__ == 'NoDefaultType'):
                result['default'] = default_val

        # Handle specific field types
        if hasattr(schema_obj, 'enum') and schema_obj.enum:
            result['enum'] = schema_obj.enum

        if hasattr(schema_obj, 'minimum') and schema_obj.minimum is not None:
            result['minimum'] = schema_obj.minimum

        if hasattr(schema_obj, 'maximum') and schema_obj.maximum is not None:
            result['maximum'] = schema_obj.maximum

        if hasattr(schema_obj, 'min_length') and schema_obj.min_length is not None:
            result['minLength'] = schema_obj.min_length

        if hasattr(schema_obj, 'max_length') and schema_obj.max_length is not None:
            result['maxLength'] = schema_obj.max_length

        if hasattr(schema_obj, 'items') and schema_obj.items:
            result['items'] = SDKDocstringExtractor._schema_to_dict(schema_obj.items)

        if hasattr(schema_obj, 'additional_properties') and schema_obj.additional_properties is not None:
            result['additionalProperties'] = SDKDocstringExtractor._schema_to_dict(schema_obj.additional_properties)

        return result

    def _extract_action_schemas(self):
        """Extract config schemas from all action classes."""
        for name in dir(ra):
            if name and name[0].isupper() and not name.startswith('_'):
                try:
                    action_class = getattr(ra, name)
                    if inspect.isclass(action_class) and hasattr(action_class, 'config_schema'):
                        schema_obj = action_class.config_schema()
                        schema_dict = self._schema_to_dict(schema_obj)
                        self._action_schemas[name] = schema_dict
                except Exception:
                    # Skip actions that can't be introspected
                    continue

    def _extract_connection_methods(self):
        """Extract docstrings from AbstractConnection methods."""
        # Get all public methods from AbstractConnection
        for name, method in inspect.getmembers(rf.AbstractConnection, predicate=inspect.isfunction):
            if not name.startswith('_'):  # Skip private methods
                if method.__doc__:
                    self._connection_methods[name] = method.__doc__.strip()

    def _extract_action_classes(self):
        """Extract docstrings from all action classes."""
        # Get all classes from rockfish.actions
        for name in dir(ra):
            # Actions are uppercase classes
            if name and name[0].isupper() and not name.startswith('_'):
                try:
                    action_class = getattr(ra, name)
                    # Verify it's a class
                    if inspect.isclass(action_class):
                        doc = action_class.__doc__
                        if doc:
                            full_doc = doc.strip()
                            self._action_classes[name] = full_doc
                            # Extract first line as summary
                            first_line = full_doc.split('\n')[0].strip()
                            self._action_summaries[name] = first_line
                except (AttributeError, TypeError):
                    continue

    def get_connection_method_doc(self, method_name: str) -> Optional[str]:
        """Get full docstring for a Connection method.

        Args:
            method_name: Name of the method (e.g., 'datasets', 'create_project')

        Returns:
            Full docstring or None if not found
        """
        return self._connection_methods.get(method_name)

    def get_action_doc(self, action_name: str) -> Optional[str]:
        """Get full docstring for an action class.

        Args:
            action_name: Name of the action class (e.g., 'TabPropertyExtractor')

        Returns:
            Full docstring or None if not found
        """
        return self._action_classes.get(action_name)

    def get_action_summary(self, action_name: str) -> Optional[str]:
        """Get one-line summary for an action class.

        Args:
            action_name: Name of the action class

        Returns:
            First line of docstring or None if not found
        """
        return self._action_summaries.get(action_name)

    def get_action_schema(self, action_name: str) -> Optional[dict]:
        """Get JSON schema for an action's configuration.

        Args:
            action_name: Name of the action class (e.g., 'TabPropertyExtractor')

        Returns:
            JSON schema dictionary or None if not found
        """
        return self._action_schemas.get(action_name)

    def get_all_action_schemas(self) -> Dict[str, dict]:
        """Get all action schemas.

        Returns:
            Dictionary mapping action names to their schemas
        """
        return self._action_schemas.copy()

    def list_all_actions(self) -> List[Tuple[str, str]]:
        """List all available action classes with summaries.

        Returns:
            List of (action_name, summary) tuples, sorted alphabetically
        """
        return sorted(
            [(name, summary) for name, summary in self._action_summaries.items()],
            key=lambda x: x[0]
        )

    def list_all_actions_with_schemas(self) -> List[Dict[str, any]]:
        """List all actions with their summaries and config schemas.

        Returns:
            List of dicts with 'name', 'summary', and 'config_schema' keys
        """
        result = []
        for name, summary in sorted(self._action_summaries.items()):
            action_info = {
                'name': name,
                'summary': summary,
                'config_schema': self._action_schemas.get(name)
            }
            result.append(action_info)
        return result

    def list_connection_methods(self) -> List[str]:
        """List all available Connection methods.

        Returns:
            Sorted list of method names
        """
        return sorted(self._connection_methods.keys())

    def format_tool_description(
        self,
        base_description: str,
        sdk_method: str,
        include_example: bool = False
    ) -> str:
        """Format an enhanced tool description with SDK info.

        Args:
            base_description: The basic tool description
            sdk_method: Name of the SDK method (e.g., 'datasets')
            include_example: Whether to include code examples

        Returns:
            Enhanced description with SDK reference
        """
        doc = self.get_connection_method_doc(sdk_method)
        if not doc:
            return base_description

        # Extract first paragraph as summary
        lines = doc.split('\n')
        summary_lines = []
        for line in lines:
            if line.strip():
                summary_lines.append(line.strip())
            elif summary_lines:
                break  # Stop at first blank line after content

        summary = ' '.join(summary_lines) if summary_lines else doc.split('\n\n')[0]

        # Build enhanced description
        enhanced = f"{base_description}\n\n"
        enhanced += f"**SDK Reference:** `Connection.{sdk_method}()`\n\n"
        enhanced += f"{summary}"

        # Extract code example if requested
        if include_example and '```' in doc:
            parts = doc.split('```')
            if len(parts) >= 2:
                example = '```' + parts[1] + '```'
                enhanced += f"\n\n**Example:**\n{example}"

        return enhanced

    def format_action_list_markdown(self) -> str:
        """Format all actions as a markdown list for documentation.

        Returns:
            Markdown-formatted string listing all actions
        """
        actions = self.list_all_actions()
        lines = ["# Rockfish SDK Actions\n"]
        lines.append(f"Total: {len(actions)} action classes available\n")

        for name, summary in actions:
            lines.append(f"- **{name}**: {summary}")

        return '\n'.join(lines)

    def format_action_doc_markdown(self, action_name: str) -> str:
        """Format action documentation as markdown.

        Args:
            action_name: Name of the action class

        Returns:
            Markdown-formatted documentation or error message
        """
        doc = self.get_action_doc(action_name)
        if not doc:
            return f"Error: Action '{action_name}' not found.\n\nAvailable actions: {', '.join(sorted(self._action_classes.keys())[:10])}..."

        # Format with markdown
        lines = [f"# {action_name}\n"]
        lines.append(doc)
        lines.append(f"\n**Usage:** `from rockfish.actions import {action_name}`")

        # Add config schema if available
        schema = self.get_action_schema(action_name)
        if schema:
            import json
            lines.append("\n## Configuration Schema\n")
            lines.append("```json")
            lines.append(json.dumps(schema, indent=2))
            lines.append("```")

            # Add a summary of required vs optional parameters
            if 'properties' in schema:
                required = schema.get('required', [])
                all_params = list(schema['properties'].keys())
                optional = [p for p in all_params if p not in required]

                lines.append(f"\n**Parameters:** {len(all_params)} total ({len(required)} required, {len(optional)} optional)")

                if required:
                    lines.append(f"\n**Required:** {', '.join(required)}")
                if optional:
                    lines.append(f"\n**Optional:** {', '.join(optional)}")

        return '\n'.join(lines)

    def format_connection_docs_markdown(self, method_name: Optional[str] = None) -> str:
        """Format Connection documentation as markdown.

        Args:
            method_name: Optional specific method name. If None, lists all methods.

        Returns:
            Markdown-formatted documentation
        """
        if method_name:
            doc = self.get_connection_method_doc(method_name)
            if not doc:
                methods = ', '.join(list(self._connection_methods.keys())[:10])
                return f"Error: Method '{method_name}' not found.\n\nAvailable methods: {methods}..."

            lines = [f"# Connection.{method_name}()\n"]
            lines.append(doc)
            return '\n'.join(lines)
        else:
            # List all methods
            lines = ["# Rockfish Connection Methods\n"]
            lines.append(f"Total: {len(self._connection_methods)} methods available\n")

            for method in sorted(self._connection_methods.keys()):
                doc = self._connection_methods[method]
                first_line = doc.split('\n')[0].strip()
                lines.append(f"- **{method}()**: {first_line}")

            return '\n'.join(lines)


# Global singleton instance
_extractor: Optional[SDKDocstringExtractor] = None


def get_extractor() -> SDKDocstringExtractor:
    """Get or create the global docstring extractor singleton.

    Returns:
        The global SDKDocstringExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = SDKDocstringExtractor()
    return _extractor
