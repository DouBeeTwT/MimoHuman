"""Tests for Agent."""

from typing import Any, AsyncGenerator

import pytest

from mimohuman.core.agent import Agent, AgentConfig
from mimohuman.core.message import AssistantMessage, Message
from mimohuman.core.provider import LLMProvider, ProviderConfig, StreamEvent, StreamEventType
from mimohuman.core.tool import Tool


class MockProvider(LLMProvider):
    """Fake provider for testing without real API calls."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config or ProviderConfig(default_model="mock"))

    def __init__(
        self,
        config: ProviderConfig | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        super().__init__(config or ProviderConfig(default_model="mock"))
        self._usage = usage or {}

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"delta": "Hello from mock!"},
        )
        yield StreamEvent(type=StreamEventType.DONE, data=dict(self._usage))

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        return AssistantMessage(content="Mock response")


@pytest.fixture
def agent() -> Agent:
    config = AgentConfig(
        name="TestAgent",
        system_prompt="You are helpful.",
        model="mock",
    )
    return Agent(config=config, provider=MockProvider())


def test_agent_creation(agent: Agent) -> None:
    assert agent.config.name == "TestAgent"
    assert agent.config.system_prompt == "You are helpful."
    assert agent.tool_registry is not None
    assert agent.hook_manager is not None


def test_agent_build_message_list(agent: Agent) -> None:
    from mimohuman.core.conversation import Conversation
    from mimohuman.core.message import UserMessage

    conv = Conversation()
    conv.add(UserMessage(content="Hello"))
    messages = agent._build_message_list(conv)

    # Should include system prompt + user message
    assert len(messages) == 2
    assert messages[0].role.value == "system"
    assert messages[1].role.value == "user"


def test_agent_config_defaults() -> None:
    config = AgentConfig()
    assert config.name == "Agent"
    assert config.max_tool_rounds == 10
    assert config.max_tokens == 4096
    assert config.temperature == 0.7


@pytest.mark.asyncio
async def test_agent_run_yields_events(agent: Agent) -> None:
    events = []
    async for event in agent.run("Hi"):
        events.append(event)

    event_types = [e.type for e in events]
    assert StreamEventType.AGENT_START in event_types
    assert StreamEventType.TEXT_DELTA in event_types
    assert StreamEventType.AGENT_END in event_types


@pytest.mark.asyncio
async def test_agent_run_includes_usage_metrics() -> None:
    provider = MockProvider(usage={"input_tokens": 50, "output_tokens": 20})
    config = AgentConfig(name="MetricsAgent", system_prompt="Test.", model="mock")
    agent = Agent(config=config, provider=provider)

    events = []
    async for event in agent.run("Hi"):
        events.append(event)

    ag_end = next(e for e in events if e.type == StreamEventType.AGENT_END)
    assert ag_end.data["input_tokens"] == 50
    assert ag_end.data["output_tokens"] == 20
    assert ag_end.data["duration_ms"] > 0
    assert ag_end.data["agent"] == "MetricsAgent"
    assert ag_end.data["rounds"] == 1
