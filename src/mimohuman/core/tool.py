"""Tool definitions and registry."""

from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from mimohuman.core.exceptions import ToolExecutionError, ToolNotFoundError


class ToolParameter(BaseModel):
    """A parameter accepted by a tool."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    items: dict[str, Any] | None = None
    enum: list[str] | None = None


class Tool(BaseModel):
    """Definition of a tool that an agent can invoke."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    handler: Callable[..., Awaitable[Any]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Generate Anthropic-compatible tool schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry for managing tool definitions and dispatching execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites any existing tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name. Raises ToolNotFoundError if not found."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_provider_schemas(self, provider_type: str = "openai") -> list[dict[str, Any]]:
        """Get tool schemas formatted for a specific provider."""
        if provider_type == "anthropic":
            return [t.to_anthropic_schema() for t in self._tools.values()]
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with the given arguments."""
        tool = self.get(name)
        if tool.handler is None:
            raise ToolExecutionError(f"Tool '{name}' has no handler")
        try:
            return await tool.handler(**kwargs)
        except Exception as e:
            raise ToolExecutionError(f"Tool '{name}' execution failed: {e}") from e
