"""Scrollable chat message list."""

from textual.containers import VerticalScroll
from textual.widgets import Static

from mimohuman.tui.widgets.markdown_content import MarkdownContent


class ChatView(VerticalScroll):
    """Scrollable container that displays chat messages."""

    def __init__(self) -> None:
        super().__init__(id="chat-view")
        self._message_count = 0

    def add_user_message(self, text: str) -> None:
        """Display a user message."""
        self._message_count += 1
        self.mount(
            Static(text, classes="chat-message user-message")
        )
        self.scroll_end(animate=False)

    def add_assistant_message(self, text: str) -> None:
        """Display a complete assistant message with markdown rendering."""
        self._message_count += 1
        self.mount(
            MarkdownContent(text, role="assistant", classes="chat-message assistant-message")
        )
        self.scroll_end(animate=False)

    def begin_streaming(self) -> MarkdownContent:
        """Create a markdown widget for a streaming assistant response.

        Returns the widget so the caller can call .set_content() on each delta.
        """
        self._message_count += 1
        widget = MarkdownContent("", role="assistant", classes="chat-message assistant-message")
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def add_system_message(self, text: str) -> None:
        """Display a system message."""
        self._message_count += 1
        self.mount(
            Static(
                f"[bold yellow]System[/bold yellow]: {text}",
                classes="chat-message system-message",
            )
        )
        self.scroll_end(animate=False)

    def add_tool_message(self, tool_name: str, args_text: str, result_text: str) -> None:
        """Display a tool call and its result."""
        self._message_count += 1
        msg = (
            f"[bold magenta]Tool: {tool_name}[/bold magenta]\n"
            f"  Args: {args_text}\n"
            f"  Result: {result_text}"
        )
        self.mount(Static(msg, classes="chat-message tool-message"))
        self.scroll_end(animate=False)

    def add_error(self, text: str) -> None:
        """Display an error message."""
        self._message_count += 1
        self.mount(
            Static(
                f"[bold red]Error[/bold red]: {text}",
                classes="chat-message error-message",
            )
        )
        self.scroll_end(animate=False)

    @property
    def message_count(self) -> int:
        return self._message_count
