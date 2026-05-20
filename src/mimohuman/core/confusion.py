"""Confusion evaluator using a Flash model in the background."""

from pathlib import Path

from mimohuman.core.message import Message, SystemMessage, UserMessage


def _load_comfuse_prompt() -> str:
    """Load the confusion evaluation prompt from prompts/comfuse.md."""
    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "comfuse.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


class ConfusionEvaluator:
    """Evaluates remaining confusion based on the full conversation.

    Uses a Flash (small/fast) model to analyze the entire conversation
    history and return a remaining-confusion percentage (0-100).

    100% = user's question completely unresolved.
    0%   = fully resolved, no remaining confusion.
    """

    def __init__(self, provider) -> None:
        self._prompt = _load_comfuse_prompt()
        self._provider = provider

    async def evaluate(
        self, conversation_messages: list[Message]
    ) -> float | None:
        """Return remaining confusion percentage, or None if evaluation failed."""
        if not self._prompt or not self._provider:
            return None

        if not conversation_messages:
            return 100.0

        conv_text = ""
        for msg in conversation_messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content[:1000]
            conv_text += f"[{role}]: {content}\n\n"

        messages = [
            SystemMessage(content=self._prompt),
            UserMessage(content=f"Conversation history:\n\n{conv_text}"),
        ]

        try:
            result = await self._provider.complete(messages=messages)
            raw = result.content.strip()
            pct = float(raw)
            return max(0.0, min(100.0, pct))
        except (ValueError, Exception):
            return None
