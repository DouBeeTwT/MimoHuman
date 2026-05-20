"""MimoHuman -- Generic AI Agent framework with TUI frontend."""

__version__ = "0.1.0"

from mimohuman.core.agent import Agent, AgentConfig
from mimohuman.core.conversation import Conversation
from mimohuman.core.exceptions import MimoHumanError
from mimohuman.core.message import (
    AssistantMessage,
    Message,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from mimohuman.core.provider import LLMProvider, ProviderConfig, StreamEvent, StreamEventType
from mimohuman.core.tool import Tool, ToolParameter, ToolRegistry

__all__ = [
    "__version__",
    # Exceptions
    "MimoHumanError",
    # Messages
    "Message",
    "MessageRole",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolResultMessage",
    "ToolCall",
    # Tools
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    # Provider
    "LLMProvider",
    "ProviderConfig",
    "StreamEvent",
    "StreamEventType",
    # Agent
    "Agent",
    "AgentConfig",
    "Conversation",
]
