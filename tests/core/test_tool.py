"""Tests for tool definitions and registry."""

import pytest

from mimohuman.core.exceptions import ToolExecutionError, ToolNotFoundError
from mimohuman.core.tool import Tool, ToolParameter, ToolRegistry


def test_tool_to_openai_schema() -> None:
    tool = Tool(
        name="search",
        description="Search the web",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query", required=True),
        ],
    )
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    params = schema["function"]["parameters"]
    assert "query" in params["properties"]
    assert "query" in params["required"]


def test_tool_to_anthropic_schema() -> None:
    tool = Tool(
        name="search",
        description="Search the web",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query", required=True),
        ],
    )
    schema = tool.to_anthropic_schema()
    assert schema["name"] == "search"
    assert "query" in schema["input_schema"]["properties"]


def test_registry_register_and_get() -> None:
    registry = ToolRegistry()
    tool = Tool(name="test", description="A test tool")
    registry.register(tool)
    assert registry.get("test") is tool


def test_registry_get_not_found() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("nonexistent")


def test_registry_unregister() -> None:
    registry = ToolRegistry()
    tool = Tool(name="test", description="Test")
    registry.register(tool)
    registry.unregister("test")
    with pytest.raises(ToolNotFoundError):
        registry.get("test")


@pytest.mark.asyncio
async def test_registry_execute() -> None:
    registry = ToolRegistry()

    async def my_handler(x: int) -> int:
        return x * 2

    tool = Tool(name="double", description="Double a number", handler=my_handler)
    registry.register(tool)

    result = await registry.execute("double", x=5)
    assert result == 10


@pytest.mark.asyncio
async def test_registry_execute_no_handler() -> None:
    registry = ToolRegistry()
    tool = Tool(name="noop", description="No handler")
    registry.register(tool)

    with pytest.raises(ToolExecutionError):
        await registry.execute("noop")
