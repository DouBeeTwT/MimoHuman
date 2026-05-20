"""Tests for the confusion evaluator."""

import pytest

from mimohuman.core.confusion import ConfusionEvaluator, _load_comfuse_prompt
from mimohuman.core.message import AssistantMessage, Message, UserMessage


class MockProviderForConfusion:
    """Provider that returns a fixed confusion score."""

    def __init__(self, score: str = "75.5"):
        self._score = score
        self._complete_calls: list = []

    async def complete(self, messages, **kwargs):
        self._complete_calls.append(messages)
        return AssistantMessage(content=self._score)


class MockProviderThatFails:
    """Provider that raises during complete."""

    async def complete(self, messages, **kwargs):
        raise RuntimeError("API failure")


def _make_conv(user: str, assistant: str) -> list[Message]:
    return [
        UserMessage(content=user),
        AssistantMessage(content=assistant),
    ]


class TestConfusionEvaluator:
    """Tests for ConfusionEvaluator."""

    def test_load_prompt_returns_string(self):
        prompt = _load_comfuse_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "confus" in prompt.lower() or "remaining" in prompt.lower()

    @pytest.mark.asyncio
    async def test_evaluate_returns_float(self):
        provider = MockProviderForConfusion("42.0")
        evaluator = ConfusionEvaluator(provider)
        conv = _make_conv("What is Python?", "Python is a language.")
        result = await evaluator.evaluate(conv)
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_evaluate_empty_conversation_returns_100(self):
        provider = MockProviderForConfusion("50.0")
        evaluator = ConfusionEvaluator(provider)
        result = await evaluator.evaluate([])
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_evaluate_clamps_to_0_100(self):
        provider = MockProviderForConfusion("-10.0")
        evaluator = ConfusionEvaluator(provider)
        result = await evaluator.evaluate(_make_conv("q", "a"))
        assert result == 0.0

        provider = MockProviderForConfusion("150.0")
        evaluator = ConfusionEvaluator(provider)
        result = await evaluator.evaluate(_make_conv("q", "a"))
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_evaluate_returns_none_on_failure(self):
        provider = MockProviderThatFails()
        evaluator = ConfusionEvaluator(provider)
        result = await evaluator.evaluate(_make_conv("q", "a"))
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_returns_none_on_invalid_response(self):
        provider = MockProviderForConfusion("not-a-number")
        evaluator = ConfusionEvaluator(provider)
        result = await evaluator.evaluate(_make_conv("q", "a"))
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_includes_conversation_content(self):
        provider = MockProviderForConfusion("50.0")
        evaluator = ConfusionEvaluator(provider)
        await evaluator.evaluate(_make_conv("QUESTION", "RESPONSE"))

        messages = provider._complete_calls[0]
        combined = "".join(m.content for m in messages)
        assert "QUESTION" in combined
        assert "RESPONSE" in combined
