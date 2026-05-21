"""Tests for the two-phase confusion evaluator."""

import json

import pytest

from mimohuman.core.confusion import (
    ConfusionEvaluator,
    InputClassification,
    InputType,
    _load_prompt,
)
from mimohuman.core.message import AssistantMessage


class MockFlashProvider:
    """Provider that returns configurable responses for different prompts."""

    def __init__(self):
        self._responses: list[str] = []
        self._call_index = 0
        self._calls: list = []

    def set_responses(self, *responses: str) -> None:
        self._responses = list(responses)
        self._call_index = 0

    async def complete(self, messages, **kwargs):
        self._calls.append(messages)
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return AssistantMessage(content=resp)
        return AssistantMessage(content='{"type": "陈述"}')


class FailingProvider:
    """Provider that raises on every call."""

    async def complete(self, messages, **kwargs):
        raise RuntimeError("API failure")


# ── InputType and InputClassification ────────────────────────────


class TestInputType:
    def test_values(self):
        assert InputType.TASK == "任务"
        assert InputType.QUESTION == "提问"
        assert InputType.STATEMENT == "陈述"


class TestInputClassification:
    def test_task_with_todolist(self):
        ic = InputClassification(
            input_type=InputType.TASK,
            todolist=["step1", "step2"],
        )
        assert ic.input_type == InputType.TASK
        assert ic.todolist == ["step1", "step2"]
        assert ic.followup_level is None
        assert ic.topic is None

    def test_question_with_followup(self):
        ic = InputClassification(
            input_type=InputType.QUESTION,
            followup_level="高",
            topic="Python",
        )
        assert ic.input_type == InputType.QUESTION
        assert ic.followup_level == "高"
        assert ic.topic == "Python"

    def test_statement(self):
        ic = InputClassification(input_type=InputType.STATEMENT)
        assert ic.input_type == InputType.STATEMENT
        assert ic.todolist is None


# ── Prompt loading ───────────────────────────────────────────────


class TestPromptLoading:
    def test_load_classify_prompt(self):
        prompt = _load_prompt("classify_input.md")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "任务" in prompt

    def test_load_todolist_prompt(self):
        prompt = _load_prompt("build_todolist.md")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_load_completion_prompt(self):
        prompt = _load_prompt("evaluate_completion.md")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_load_followup_prompt(self):
        prompt = _load_prompt("predict_followup.md")
        assert isinstance(prompt, str)
        assert "高" in prompt

    def test_load_nonexistent_returns_empty(self):
        prompt = _load_prompt("nonexistent_prompt.md")
        assert prompt == ""


# ── classify_input ───────────────────────────────────────────────


class TestClassifyInput:
    @pytest.mark.asyncio
    async def test_classify_task(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '["Step 1", "Step 2", "Step 3"]',
        )
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("帮我写一个排序算法")
        assert result is not None
        assert result.input_type == InputType.TASK
        assert result.todolist == ["Step 1", "Step 2", "Step 3"]
        assert evaluator._current_input_type == InputType.TASK
        assert evaluator._current_todolist == ["Step 1", "Step 2", "Step 3"]

    @pytest.mark.asyncio
    async def test_classify_question(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "高", "topic": "机器学习"}',
        )
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("什么是机器学习？")
        assert result is not None
        assert result.input_type == InputType.QUESTION
        assert result.followup_level == "高"
        assert result.topic == "机器学习"
        assert evaluator._last_followup_level == "高"
        assert evaluator._last_topic == "机器学习"

    @pytest.mark.asyncio
    async def test_classify_statement(self):
        provider = MockFlashProvider()
        provider.set_responses('{"type": "陈述"}')
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("你好")
        assert result is not None
        assert result.input_type == InputType.STATEMENT
        assert result.todolist is None
        assert result.followup_level is None

    @pytest.mark.asyncio
    async def test_classify_returns_none_on_failure(self):
        provider = FailingProvider()
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_classify_returns_none_on_invalid_json(self):
        provider = MockFlashProvider()
        provider.set_responses("not json")
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_classify_returns_none_on_invalid_type(self):
        provider = MockFlashProvider()
        provider.set_responses('{"type": "unknown"}')
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_classify_task_with_todolist_failure(self):
        """If todolist generation fails, classification still succeeds."""
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            "not json",
        )
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("do something")
        assert result is not None
        assert result.input_type == InputType.TASK
        assert result.todolist is None


# ── compute: 任务 ────────────────────────────────────────────────


class TestComputeTask:
    @pytest.mark.asyncio
    async def test_task_full_completion(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '["Step 1", "Step 2"]',
            "1.0",
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("do something")
        cfs = await evaluator.compute("done", output_tokens=100)
        # uncompleted = 0, token_reduction = 100/2048 ≈ 0.049
        # confusion = max(0, 0 - 0.049) = 0.0
        assert cfs == 0.0

    @pytest.mark.asyncio
    async def test_task_no_completion(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '["Step 1", "Step 2"]',
            "0.0",
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("do something")
        cfs = await evaluator.compute("irrelevant", output_tokens=0)
        # uncompleted = 1.0, token_reduction = 0
        # confusion = max(0, 1.0 - 0) = 1.0
        assert cfs == 1.0

    @pytest.mark.asyncio
    async def test_task_partial_completion_with_token_reduction(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '["Step 1", "Step 2", "Step 3", "Step 4"]',
            "0.5",
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("do something")
        cfs = await evaluator.compute("partial", output_tokens=512)
        # uncompleted = 0.5, token_reduction = 512/2048 = 0.25
        # confusion = max(0, 0.5 - 0.25) = 0.25
        assert abs(cfs - 0.25) < 0.001

    @pytest.mark.asyncio
    async def test_task_clamps_to_zero(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '["Step 1"]',
            "0.8",
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("do something")
        cfs = await evaluator.compute("great response", output_tokens=4096)
        # uncompleted = 0.2, token_reduction = 4096/2048 = 2.0
        # confusion = max(0, 0.2 - 2.0) = 0.0
        assert cfs == 0.0

    @pytest.mark.asyncio
    async def test_task_with_no_todolist_keeps_confusion(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            "invalid json",  # todolist fails
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("do something")
        cfs = await evaluator.compute("response", output_tokens=100)
        # No todolist, so confusion stays at current value (1.0)
        assert cfs == 1.0

    @pytest.mark.asyncio
    async def test_task_completion_eval_failure_defaults_to_zero(self):
        """If completion evaluation fails, assume 0% completion."""
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '["Step 1"]',
            "not a number",
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("do something")
        cfs = await evaluator.compute("response", output_tokens=0)
        # completion_ratio defaults to 0.0 on failure
        # uncompleted = 1.0, confusion = max(0, 1.0 - 0) = 1.0
        assert cfs == 1.0


# ── compute: 提问 ────────────────────────────────────────────────


class TestComputeQuestion:
    @pytest.mark.asyncio
    async def test_question_reduces_by_tokens(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "中", "topic": "Python"}',
        )
        evaluator = ConfusionEvaluator(provider)
        evaluator._confusion = 0.8

        await evaluator.classify_input("什么是Python？")
        cfs = await evaluator.compute("Python is ...", output_tokens=2048)
        # confusion = max(0, 0.8 - 2048/2048) = max(0, 0.8 - 1.0) = 0.0
        assert cfs == 0.0

    @pytest.mark.asyncio
    async def test_question_reduces_by_tokens_partial(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "中", "topic": "Python"}',
        )
        evaluator = ConfusionEvaluator(provider)
        evaluator._confusion = 0.8

        await evaluator.classify_input("什么是Python？")
        cfs = await evaluator.compute("Python is ...", output_tokens=512)
        # confusion = max(0, 0.8 - 512/2048) = max(0, 0.8 - 0.25) = 0.55
        assert abs(cfs - 0.55) < 0.001

    @pytest.mark.asyncio
    async def test_question_clamps_to_zero(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "低", "topic": "math"}',
        )
        evaluator = ConfusionEvaluator(provider)
        evaluator._confusion = 0.05

        await evaluator.classify_input("1+1=?")
        cfs = await evaluator.compute("2", output_tokens=20480)
        # confusion = max(0, 0.05 - 10.0) = 0.0
        assert cfs == 0.0

    @pytest.mark.asyncio
    async def test_followup_same_topic_reduces_confusion(self):
        provider = MockFlashProvider()
        # First question
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "高", "topic": "机器学习 方法"}',
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("机器学习有哪些方法？")
        cfs1 = await evaluator.compute("有监督、无监督...", output_tokens=0)
        # confusion = max(0, 1.0 - 0) = 1.0
        assert cfs1 == 1.0
        assert evaluator._last_followup_level == "高"
        assert evaluator._last_topic == "机器学习 方法"

        # Second question on same topic
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "中", "topic": "机器学习 深度学习"}',
        )
        await evaluator.classify_input("深度学习和机器学习的关系？")
        # Same topic detected (机器学习 overlaps)
        # Previous level was 高, reduction = 0.15
        # confusion = max(0, 1.0 - 0.15) = 0.85
        assert evaluator._confusion == 0.85

        cfs2 = await evaluator.compute("深度学习是机器学习的子集", output_tokens=1024)
        # confusion = max(0, 0.85 - 1024/2048) = max(0, 0.85 - 0.5) = 0.35
        assert abs(cfs2 - 0.35) < 0.001

    @pytest.mark.asyncio
    async def test_followup_different_topic_no_reduction(self):
        provider = MockFlashProvider()
        # First question
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "高", "topic": "Python"}',
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("什么是Python？")
        await evaluator.compute("Python is ...", output_tokens=0)

        # Second question on different topic
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "中", "topic": "天气"}',
        )
        await evaluator.classify_input("今天天气怎么样？")
        # Different topic, no follow-up reduction
        assert evaluator._confusion == 1.0

    @pytest.mark.asyncio
    async def test_followup_reduction_medium(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "中", "topic": "API"}',
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("什么是REST API？")
        await evaluator.compute("REST is ...", output_tokens=0)

        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "低", "topic": "API 设计"}',
        )
        await evaluator.classify_input("如何设计好的API？")
        # Previous level 中, reduction = 0.10
        assert evaluator._confusion == 0.90

    @pytest.mark.asyncio
    async def test_followup_reduction_low(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "低", "topic": "数学"}',
        )
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("1+1=?")
        await evaluator.compute("2", output_tokens=0)

        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "低", "topic": "数学 计算"}',
        )
        await evaluator.classify_input("那2+2呢？")
        # Previous level 低, reduction = 0.05
        assert evaluator._confusion == 0.95

    @pytest.mark.asyncio
    async def test_question_with_followup_failure_keeps_confusion(self):
        """If follow-up prediction fails, no reduction but confusion still works."""
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            "not json",  # followup fails
        )
        evaluator = ConfusionEvaluator(provider)
        evaluator._confusion = 0.8

        result = await evaluator.classify_input("what?")
        assert result is not None
        assert result.input_type == InputType.QUESTION
        assert result.followup_level is None

        cfs = await evaluator.compute("answer", output_tokens=0)
        assert cfs == 0.8


# ── compute: 陈述 ────────────────────────────────────────────────


class TestComputeStatement:
    @pytest.mark.asyncio
    async def test_statement_sets_confusion_to_zero(self):
        provider = MockFlashProvider()
        provider.set_responses('{"type": "陈述"}')
        evaluator = ConfusionEvaluator(provider)
        evaluator._confusion = 0.75

        await evaluator.classify_input("谢谢")
        cfs = await evaluator.compute("不客气", output_tokens=100)
        assert cfs == 0.0

    @pytest.mark.asyncio
    async def test_statement_resets_from_any_value(self):
        provider = MockFlashProvider()
        provider.set_responses('{"type": "陈述"}')
        evaluator = ConfusionEvaluator(provider)

        await evaluator.classify_input("你好")
        cfs = await evaluator.compute("你好！", output_tokens=0)
        assert cfs == 0.0


# ── State persistence across turns ──────────────────────────────


class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_confusion_persists_across_turns(self):
        provider = MockFlashProvider()
        evaluator = ConfusionEvaluator(provider)
        assert evaluator.confusion == 1.0

        # Turn 1: statement -> 0
        provider.set_responses('{"type": "陈述"}')
        await evaluator.classify_input("你好")
        cfs = await evaluator.compute("你好", output_tokens=0)
        assert cfs == 0.0
        assert evaluator.confusion == 0.0

    @pytest.mark.asyncio
    async def test_state_reset_between_turns(self):
        provider = MockFlashProvider()
        evaluator = ConfusionEvaluator(provider)

        # Turn 1: task
        provider.set_responses(
            '{"type": "任务"}',
            '["step1"]',
            "1.0",
        )
        await evaluator.classify_input("do X")
        await evaluator.compute("done", output_tokens=0)

        # Turn 2: verify per-turn state was reset
        assert evaluator._current_input_type is None
        assert evaluator._current_todolist is None
        assert evaluator._current_followup_level is None

    @pytest.mark.asyncio
    async def test_compute_without_classify_returns_current(self):
        provider = MockFlashProvider()
        evaluator = ConfusionEvaluator(provider)
        evaluator._confusion = 0.42

        # Call compute without classify_input first
        cfs = await evaluator.compute("response", output_tokens=100)
        assert cfs == 0.42

    @pytest.mark.asyncio
    async def test_followup_state_persists_across_question_turns(self):
        provider = MockFlashProvider()
        evaluator = ConfusionEvaluator(provider)

        # Turn 1: question
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "高", "topic": "AI"}',
        )
        await evaluator.classify_input("什么是AI？")
        await evaluator.compute("AI is ...", output_tokens=0)
        assert evaluator._last_followup_level == "高"
        assert evaluator._last_topic == "AI"

        # Turn 2: statement clears no state but doesn't affect followup tracking
        provider.set_responses('{"type": "陈述"}')
        await evaluator.classify_input("知道了")
        await evaluator.compute("好的", output_tokens=0)

        # The last followup info should still be there
        assert evaluator._last_followup_level == "高"
        assert evaluator._last_topic == "AI"

    @pytest.mark.asyncio
    async def test_topic_matching(self):
        assert ConfusionEvaluator._topics_match("机器学习 方法", "机器学习 深度学习") is True
        assert ConfusionEvaluator._topics_match("Python", "天气") is False
        assert ConfusionEvaluator._topics_match("", "topic") is False
        assert ConfusionEvaluator._topics_match("topic", "") is False
        assert ConfusionEvaluator._topics_match("Python 编程", "Python 入门") is True


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_provider(self):
        evaluator = ConfusionEvaluator(None)
        result = await evaluator.classify_input("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_todolist_from_model(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '[]',
        )
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("do something")
        assert result is not None
        assert result.todolist == []

    @pytest.mark.asyncio
    async def test_todolist_with_non_string_items(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "任务"}',
            '[1, 2, 3]',
        )
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("do something")
        assert result is not None
        # Non-string items should cause todolist to be None
        assert result.todolist is None

    @pytest.mark.asyncio
    async def test_followup_level_out_of_range(self):
        provider = MockFlashProvider()
        provider.set_responses(
            '{"type": "提问"}',
            '{"level": "extreme", "topic": "test"}',
        )
        evaluator = ConfusionEvaluator(provider)

        result = await evaluator.classify_input("question?")
        assert result is not None
        assert result.followup_level is None
        assert result.topic is None
