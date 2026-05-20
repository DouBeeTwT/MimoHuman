"""Collapsible thinking/reasoning display widget."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Collapsible, Static


class ThinkingDisplay(Vertical):
    """A collapsible section showing the agent's thinking process.

    Auto-collapses when the final answer arrives.
    """

    def __init__(self) -> None:
        super().__init__(id="thinking-display")
        self._content = ""

    def compose(self) -> ComposeResult:
        with Collapsible(title="Thinking...", collapsed_symbol="▶", expanded_symbol="▼"):
            yield Static(self._content, id="thinking-text")

    def append(self, delta: str) -> None:
        """Append a chunk of thinking content."""
        self._content += delta
        text_widget = self.query_one("#thinking-text", Static)
        text_widget.update(self._content)

    def get_content(self) -> str:
        return self._content
