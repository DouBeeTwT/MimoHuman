"""Example: Launch the TUI with a mock agent for testing."""

import asyncio
import os

from mimohuman.core.agent import Agent, AgentConfig
from mimohuman.core.provider import ProviderConfig
from mimohuman.core.tool import Tool, ToolParameter, ToolRegistry
from mimohuman.providers.anthropic_provider import AnthropicProvider
from mimohuman.tui.app import MimoHumanApp
from mimohuman.tui.controller import TUIController


async def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Expression contains disallowed characters"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    # Set up tools
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a mathematical expression",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Math expression to evaluate (e.g., '2 + 2')",
                    required=True,
                ),
            ],
            handler=calculator,
        )
    )

    # Configure
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("MIMOHUMAN_MODEL", "claude-sonnet-4-6")

    agent_config = AgentConfig(
        name="MimoHuman Demo",
        system_prompt="You are a helpful assistant with a calculator tool.",
        model=model,
    )
    provider_config = ProviderConfig(
        api_key=api_key,
        default_model=model,
    )
    provider = AnthropicProvider(provider_config)
    agent = Agent(config=agent_config, provider=provider, tool_registry=registry)
    controller = TUIController(agent)

    app = MimoHumanApp(controller)
    app.run()


if __name__ == "__main__":
    main()
