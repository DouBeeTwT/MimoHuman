"""Message types -- the universal data currency of MimoHuman."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Base message model. Use the typed subclasses."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_provider_format(self, provider_type: str = "openai") -> dict[str, Any]:
        """Convert this message to a provider-compatible dict.

        Override in subclasses for provider-specific formatting.
        """
        raise NotImplementedError("Use a concrete message subclass")


class SystemMessage(Message):
    """A system prompt message."""

    role: MessageRole = MessageRole.SYSTEM

    def to_provider_format(self, provider_type: str = "openai") -> dict[str, Any]:
        return {"role": "system", "content": self.content}


class UserMessage(Message):
    """A user message, optionally with attachments."""

    role: MessageRole = MessageRole.USER
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    def to_provider_format(self, provider_type: str = "openai") -> dict[str, Any]:
        if provider_type == "anthropic":
            content: list[dict[str, Any]] = [{"type": "text", "text": self.content}]
            for att in self.attachments:
                content.append(att)
            return {"role": "user", "content": content}
        return {"role": "user", "content": self.content}


class AssistantMessage(Message):
    """An assistant response, possibly containing tool calls."""

    role: MessageRole = MessageRole.ASSISTANT
    tool_calls: list[ToolCall] = Field(default_factory=list)
    thinking: str | None = None
    finish_reason: str | None = None

    def to_provider_format(self, provider_type: str = "openai") -> dict[str, Any]:
        result: dict[str, Any] = {"role": "assistant"}
        if self.content:
            result["content"] = self.content
        if self.tool_calls and provider_type == "openai":
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.name},
                }
                for tc in self.tool_calls
            ]
        elif self.tool_calls and provider_type == "anthropic":
            result["content"] = [
                {"type": "text", "text": self.content},
            ] + [
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
                for tc in self.tool_calls
            ]
        return result


class ToolResultMessage(Message):
    """Result of a tool execution."""

    role: MessageRole = MessageRole.TOOL
    tool_call_id: str = ""
    name: str = ""
    is_error: bool = False

    def to_provider_format(self, provider_type: str = "openai") -> dict[str, Any]:
        if provider_type == "anthropic":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id,
                        "content": self.content,
                        "is_error": self.is_error,
                    }
                ],
            }
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
