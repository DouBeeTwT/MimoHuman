"""Two-phase confusion evaluator using a Flash model.

Phase 1: classify_input() — before Pro model responds
  - Classifies user input as 任务/提问/陈述
  - For 任务: generates a TODO list
  - For 提问: predicts follow-up likelihood and extracts topic

Phase 2: compute() — after Pro model responds
  - Computes confusion score based on input type and Pro model output
"""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from mimohuman.core.message import SystemMessage, UserMessage

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt file from the prompts/ directory."""
    path = _PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


class InputType(str, Enum):
    """Classification of user input."""

    TASK = "任务"
    QUESTION = "提问"
    STATEMENT = "陈述"


class InputClassification(BaseModel):
    """Result of Phase 1 classification."""

    input_type: InputType
    todolist: list[str] | None = None
    followup_level: str | None = None  # "高" / "中" / "低"
    topic: str | None = None


# Follow-up reduction ratios by predicted level (0~1 range)
_FOLLOWUP_REDUCTION = {"高": 0.15, "中": 0.10, "低": 0.05}


class ConfusionEvaluator:
    """Two-phase confusion evaluator.

    Phase 1 (before Pro response): classify_input() -> InputClassification
    Phase 2 (after Pro response):  compute() -> float
    """

    def __init__(self, provider) -> None:
        self._provider = provider
        self._classify_prompt = _load_prompt("classify_input.md")
        self._todolist_prompt = _load_prompt("build_todolist.md")
        self._completion_prompt = _load_prompt("evaluate_completion.md")
        self._followup_prompt = _load_prompt("predict_followup.md")

        # Persistent state across turns (0.0~1.0 range)
        self._confusion: float = 1.0
        self._last_followup_level: str | None = None
        self._last_topic: str | None = None

        # Per-turn state (set by classify_input, consumed by compute)
        self._current_todolist: list[str] | None = None
        self._current_input_type: InputType | None = None
        self._current_followup_level: str | None = None

    @property
    def confusion(self) -> float:
        """Current confusion score."""
        return self._confusion

    async def classify_input(self, user_text: str) -> InputClassification | None:
        """Phase 1: Classify user input and prepare for computation.

        Returns InputClassification on success, None on failure.
        """
        if not self._provider or not self._classify_prompt:
            return None

        self._current_todolist = None
        self._current_followup_level = None

        # Step 1: Classify input type
        classification = await self._classify(user_text)
        if classification is None:
            return None

        self._current_input_type = classification.input_type

        if classification.input_type == InputType.TASK:
            # Step 2a: Build TODO list for tasks
            todolist = await self._build_todolist(user_text)
            classification.todolist = todolist
            self._current_todolist = todolist

        elif classification.input_type == InputType.QUESTION:
            # Step 2b: Predict follow-up for questions
            level, topic = await self._predict_followup(user_text)
            classification.followup_level = level
            classification.topic = topic
            self._current_followup_level = level

            # Check if this is a follow-up to previous question on same topic
            if (
                self._last_followup_level is not None
                and self._last_topic is not None
                and topic
                and self._topics_match(self._last_topic, topic)
            ):
                reduction = _FOLLOWUP_REDUCTION.get(self._last_followup_level, 0.0)
                self._confusion = max(0.0, self._confusion - reduction)

            # Store for next turn's follow-up detection
            self._last_followup_level = level
            self._last_topic = topic

        return classification

    async def compute(self, output_text: str, output_tokens: int) -> float:
        """Phase 2: Compute confusion score after Pro model responds.

        Returns updated confusion ratio (0.0 - 1.0). Multiply by 100 for display.
        """
        if self._current_input_type is None:
            return self._confusion

        if self._current_input_type == InputType.TASK:
            self._confusion = await self._compute_task(output_text, output_tokens)
        elif self._current_input_type == InputType.QUESTION:
            self._confusion = self._compute_question(output_tokens)
        elif self._current_input_type == InputType.STATEMENT:
            self._confusion = 0.0

        # Reset per-turn state
        self._current_input_type = None
        self._current_todolist = None
        self._current_followup_level = None

        return self._confusion

    # ── Internal methods ──────────────────────────────────────────

    async def _classify(self, user_text: str) -> InputClassification | None:
        """Call Flash model to classify user input type."""
        messages = [
            SystemMessage(content=self._classify_prompt),
            UserMessage(content=user_text),
        ]
        try:
            result = await self._provider.complete(messages=messages)
            raw = result.content.strip()
            data = json.loads(raw)
            input_type = InputType(data["type"])
            return InputClassification(input_type=input_type)
        except (json.JSONDecodeError, KeyError, ValueError, Exception):
            return None

    async def _build_todolist(self, user_text: str) -> list[str] | None:
        """Call Flash model to generate a TODO list for a task."""
        if not self._todolist_prompt:
            return None
        messages = [
            SystemMessage(content=self._todolist_prompt),
            UserMessage(content=user_text),
        ]
        try:
            result = await self._provider.complete(messages=messages)
            raw = result.content.strip()
            items = json.loads(raw)
            if isinstance(items, list) and all(isinstance(s, str) for s in items):
                return items
            return None
        except (json.JSONDecodeError, ValueError, Exception):
            return None

    async def _predict_followup(self, user_text: str) -> tuple[str | None, str | None]:
        """Call Flash model to predict follow-up likelihood and extract topic."""
        if not self._followup_prompt:
            return None, None
        messages = [
            SystemMessage(content=self._followup_prompt),
            UserMessage(content=user_text),
        ]
        try:
            result = await self._provider.complete(messages=messages)
            raw = result.content.strip()
            data = json.loads(raw)
            level = data.get("level")
            topic = data.get("topic")
            if level in ("高", "中", "低"):
                return level, topic
            return None, None
        except (json.JSONDecodeError, KeyError, ValueError, Exception):
            return None, None

    async def _compute_task(self, output_text: str, output_tokens: int) -> float:
        """Compute confusion for a task input type.

        confusion = max(0.0, uncompleted_ratio - output_tokens / 2048)
        """
        if not self._current_todolist or not self._completion_prompt:
            return self._confusion

        # Ask Flash model to evaluate TODO list completion
        todo_text = json.dumps(self._current_todolist, ensure_ascii=False)
        user_content = f"## TODO List\n{todo_text}\n\n## Assistant Response\n{output_text}"
        messages = [
            SystemMessage(content=self._completion_prompt),
            UserMessage(content=user_content),
        ]
        try:
            result = await self._provider.complete(messages=messages)
            raw = result.content.strip()
            completion_ratio = float(raw)
            completion_ratio = max(0.0, min(1.0, completion_ratio))
        except (ValueError, Exception):
            completion_ratio = 0.0

        uncompleted_ratio = 1.0 - completion_ratio
        token_reduction = output_tokens / 2048.0
        return max(0.0, uncompleted_ratio - token_reduction)

    def _compute_question(self, output_tokens: int) -> float:
        """Compute confusion for a question input type.

        confusion -= output_tokens / 2048
        """
        token_reduction = output_tokens / 2048.0
        return max(0.0, self._confusion - token_reduction)

    @staticmethod
    def _topics_match(topic_a: str, topic_b: str) -> bool:
        """Check if two topics are related (simple overlap check)."""
        if not topic_a or not topic_b:
            return False
        a_words = set(topic_a.lower().split())
        b_words = set(topic_b.lower().split())
        if not a_words or not b_words:
            return topic_a.lower() == topic_b.lower()
        overlap = a_words & b_words
        return len(overlap) > 0
