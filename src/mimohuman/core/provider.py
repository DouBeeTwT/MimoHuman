"""LLM Provider abstraction and event types."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

from mimohuman.core.message import AssistantMessage, Message
from mimohuman.core.tool import Tool


class StreamEventType(str, Enum):
    """Types of streaming events emitted by providers."""

    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    DONE = "done"
    ERROR = "error"

    # Agent-level events (extended by Agent)
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_RESULT = "tool_result"


class StreamEvent(BaseModel):
    """A single event in a streaming response.

    Uses a flat event type + data dict design so TUI code can
    pattern-match cleanly on event.type.
    """

    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_key: str = ""
    base_url: str | None = None
    default_model: str = ""
    extra_headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 120.0
    max_retries: int = 3


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Each concrete provider (Anthropic, OpenAI, etc.) translates its
    native streaming chunks into typed StreamEvent objects so the Agent
    and TUI consume a uniform event stream.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response for the given messages."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        """Non-streaming completion for the given messages."""
        ...

    def supports_thinking(self) -> bool:
        """Whether this provider supports extended thinking/reasoning."""
        return False

    def supports_vision(self) -> bool:
        """Whether this provider supports image inputs."""
        return False

    @property
    def default_model(self) -> str:
        return self.config.default_model
