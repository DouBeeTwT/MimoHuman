"""Shared test fixtures."""

import pytest

from mimohuman.core.agent import Agent, AgentConfig
from mimohuman.core.conversation import Conversation
from mimohuman.core.provider import ProviderConfig
from mimohuman.core.tool import Tool, ToolRegistry
from mimohuman.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        name="TestAgent",
        system_prompt="You are a test agent.",
        model="claude-sonnet-4-6",
    )


@pytest.fixture
def provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test-key",
        default_model="claude-sonnet-4-6",
    )


@pytest.fixture
def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    return registry


@pytest.fixture
def conversation() -> Conversation:
    return Conversation()
