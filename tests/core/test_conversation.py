"""Tests for Conversation."""

from mimohuman.core.conversation import Conversation
from mimohuman.core.message import UserMessage


def test_conversation_add_message() -> None:
    conv = Conversation()
    msg = UserMessage(content="Hello")
    conv.add(msg)
    assert len(conv) == 1


def test_conversation_get_context_window() -> None:
    conv = Conversation()
    for i in range(10):
        conv.add(UserMessage(content=f"Message {i}"))

    window = conv.get_context_window(max_tokens=10)
    assert len(window) <= 10


def test_conversation_clear() -> None:
    conv = Conversation()
    conv.add(UserMessage(content="Hello"))
    conv.clear()
    assert len(conv) == 0


def test_conversation_clone() -> None:
    conv = Conversation()
    conv.add(UserMessage(content="Original"))
    cloned = conv.clone()

    cloned.add(UserMessage(content="Clone only"))
    assert len(conv) == 1
    assert len(cloned) == 2


def test_conversation_serialization() -> None:
    conv = Conversation()
    conv.add(UserMessage(content="Hello"))
    data = conv.to_dict()
    restored = Conversation.from_dict(data)
    assert len(restored) == 1
    assert restored.messages[0].content == "Hello"
