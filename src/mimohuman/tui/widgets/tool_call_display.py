"""Tool call visualization widget."""

import json

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static


class ToolCallDisplay(Vertical):
    """Shows a single tool call with its status, arguments, and result.

    States: pending -> running -> done (success/error)
    """

    def __init__(self, tool_id: str, tool_name: str) -> None:
        super().__init__(id=f"tool-{tool_id}", classes="tool-call")
        self.tool_id = tool_id
        self.tool_name = tool_name
        self._arguments = ""
        self._result = ""
        self._state = "pending"

    def compose(self) -> ComposeResult:
        icon = {"pending": "⏳", "running": "🔄", "success": "✅", "error": "❌"}.get(
            self._state, "⏳"
        )
        with Collapsible(
            title=f"{icon} Tool: {self.tool_name}",
            collapsed_symbol="▶",
            expanded_symbol="▼",
        ):
            yield Static(self._arguments or "(no arguments)", id="tool-args")
            yield Static("", id="tool-result")

    def set_arguments(self, args_json: str) -> None:
        """Set the tool call arguments."""
        self._arguments = args_json
        try:
            formatted = json.dumps(json.loads(args_json), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            formatted = args_json
        self._update_section("tool-args", f"Arguments:\n```json\n{formatted}\n```")

    def set_result(self, result: str, is_error: bool = False) -> None:
        """Set the tool execution result."""
        self._result = result
        self._state = "error" if is_error else "success"
        self._update_section(
            "tool-result",
            f"{'Error' if is_error else 'Result'}:\n```\n{result}\n```",
        )

    def set_running(self) -> None:
        """Mark the tool as currently executing."""
        self._state = "running"

    def _update_section(self, widget_id: str, text: str) -> None:
        """Safely update a child widget by id."""
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(text)
        except Exception:
            pass
