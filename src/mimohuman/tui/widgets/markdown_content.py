"""Markdown rendering widget."""

from textual.widgets import Static


class MarkdownContent(Static):
    """Renders markdown content using Rich's Markdown renderable."""

    def __init__(self, content: str = "", role: str = "assistant", **kwargs) -> None:
        super().__init__("", **kwargs)
        self._role = role
        self.set_content(content)

    def set_content(self, content: str) -> None:
        """Update the displayed markdown content."""
        try:
            from rich.markdown import Markdown

            self.update(Markdown(content))
        except ImportError:
            self.update(content)
