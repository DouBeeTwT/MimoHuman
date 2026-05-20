"""Conversation -- ordered message history with context window support."""

import uuid
from typing import Any

from pydantic import BaseModel, Field

from mimohuman.core.message import Message


class Conversation(BaseModel):
    """Ordered container for conversation messages.

    Supports context-window truncation so long conversations don't
    exceed the model's token limit.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add(self, message: Message) -> None:
        """Append a message to the conversation."""
        self.messages.append(message)

    def get_messages(self) -> list[Message]:
        """Return all messages in order."""
        return list(self.messages)

    def get_context_window(self, max_tokens: int) -> list[Message]:
        """Return as many recent messages as fit within max_tokens.

        Uses a simple char/4 heuristic for token estimation.
        Replaceable with tiktoken later.
        """
        estimated = 0
        window: list[Message] = []
        for msg in reversed(self.messages):
            estimated += len(msg.content) // 4 + 1
            if estimated > max_tokens:
                break
            window.append(msg)
        return list(reversed(window))

    def clear(self) -> None:
        """Remove all messages."""
        self.messages.clear()

    def clone(self) -> "Conversation":
        """Deep-copy this conversation."""
        return self.model_copy(deep=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Deserialize from a plain dict."""
        return cls.model_validate(data)

    def __len__(self) -> int:
        return len(self.messages)

    def __bool__(self) -> bool:
        return len(self.messages) > 0
