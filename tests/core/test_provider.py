"""Tests for provider abstraction."""

from mimohuman.core.provider import ProviderConfig, StreamEvent, StreamEventType


def test_provider_config_defaults() -> None:
    config = ProviderConfig(api_key="sk-test", default_model="gpt-4")
    assert config.api_key == "sk-test"
    assert config.default_model == "gpt-4"
    assert config.timeout == 120.0
    assert config.max_retries == 3


def test_stream_event_creation() -> None:
    event = StreamEvent(
        type=StreamEventType.TEXT_DELTA,
        data={"delta": "Hello"},
    )
    assert event.type == StreamEventType.TEXT_DELTA
    assert event.data["delta"] == "Hello"


def test_stream_event_types() -> None:
    """Verify all event types are accessible."""
    assert StreamEventType.TEXT_DELTA == "text_delta"
    assert StreamEventType.THINKING_DELTA == "thinking_delta"
    assert StreamEventType.TOOL_CALL_START == "tool_call_start"
    assert StreamEventType.DONE == "done"
    assert StreamEventType.ERROR == "error"
