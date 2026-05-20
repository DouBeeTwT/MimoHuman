"""Example: Basic agent with a custom tool."""

import asyncio
import os

from mimohuman.core.agent import Agent, AgentConfig
from mimohuman.core.provider import ProviderConfig, StreamEventType
from mimohuman.core.tool import Tool, ToolParameter, ToolRegistry
from mimohuman.providers.anthropic_provider import AnthropicProvider


async def search_tool(query: str) -> str:
    """Mock search tool."""
    return f"Results for '{query}': No real results (mock tool)."


async def main() -> None:
    # Set up tool registry
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="search",
            description="Search the web for information",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query",
                    required=True,
                ),
            ],
            handler=search_tool,
        )
    )

    # Configure provider
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    provider_config = ProviderConfig(
        api_key=api_key,
        default_model="claude-sonnet-4-6",
    )
    provider = AnthropicProvider(provider_config)

    # Configure agent
    agent_config = AgentConfig(
        name="SearchAgent",
        system_prompt="You are a helpful assistant with access to a search tool.",
        model="claude-sonnet-4-6",
    )

    agent = Agent(
        config=agent_config,
        provider=provider,
        tool_registry=registry,
    )

    # Run the agent
    print("=" * 50)
    print("MimoHuman Basic Agent Example")
    print("=" * 50)

    async for event in agent.run("What is the weather in Tokyo today?"):
        match event.type:
            case StreamEventType.TEXT_DELTA:
                print(event.data["delta"], end="", flush=True)
            case StreamEventType.TOOL_CALL_START:
                print(f"\n[Calling tool: {event.data['name']}]")
            case StreamEventType.TOOL_RESULT:
                print(f"[Tool result: {event.data['result'][:100]}]")
            case StreamEventType.ERROR:
                print(f"\n[Error: {event.data.get('message')}]")
            case StreamEventType.AGENT_END:
                print(f"\n\n(Agent finished in {event.data.get('rounds', 1)} rounds)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
