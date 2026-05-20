"""Tests for Anthropic provider."""

import pytest

from mimohuman.core.provider import ProviderConfig

anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")


def test_anthropic_provider_creation() -> None:
    from mimohuman.providers.anthropic_provider import AnthropicProvider

    config = ProviderConfig(
        api_key="test-key",
        default_model="claude-sonnet-4-6",
    )
    provider = AnthropicProvider(config)
    assert provider.supports_thinking() is True
    assert provider.supports_vision() is True
    assert provider.default_model == "claude-sonnet-4-6"


def test_anthropic_provider_requires_api_key() -> None:
    from mimohuman.providers.anthropic_provider import AnthropicProvider

    config = ProviderConfig(default_model="claude-sonnet-4-6")
    provider = AnthropicProvider(config)
    assert provider.config.api_key == ""
