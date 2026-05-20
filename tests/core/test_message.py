"""Tests for message models."""

from mimohuman.core.message import (
    AssistantMessage,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def test_user_message_creation() -> None:
    msg = UserMessage(content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.timestamp is not None


def test_system_message_to_provider_format() -> None:
    msg = SystemMessage(content="You are helpful.")
    formatted = msg.to_provider_format("openai")
    assert formatted == {"role": "system", "content": "You are helpful."}


def test_tool_result_message_to_provider_format() -> None:
    msg = ToolResultMessage(
        content="42",
        tool_call_id="call_1",
        name="calculator",
        is_error=False,
    )
    formatted = msg.to_provider_format("openai")
    assert formatted["role"] == "tool"
    assert formatted["tool_call_id"] == "call_1"


def test_assistant_message_with_tool_calls() -> None:
    msg = AssistantMessage(
        content="Let me search.",
        tool_calls=[
            ToolCall(id="tc1", name="search", arguments={"query": "weather"})
        ],
    )
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"


def test_message_timestamp_auto_generated() -> None:
    msg = UserMessage(content="Hi")
    assert msg.timestamp is not None
