"""Input bar with history navigation."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Input


class InputBar(Horizontal):
    """Bottom input bar with text input, history, and send button."""

    class SendMessage(Message):
        """Posted when the user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self) -> None:
        super().__init__(id="input-bar")
        self._history: list[str] = []
        self._history_index = -1

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a message...", id="prompt-input")
        yield Button("Send", id="send-button")

    @on(Button.Pressed, "#send-button")
    def on_send_button(self) -> None:
        self._submit()

    @on(Input.Submitted, "#prompt-input")
    def on_input_submitted(self) -> None:
        self._submit()

    def _submit(self) -> None:
        inp = self.query_one("#prompt-input", Input)
        text = inp.value.strip()
        if not text:
            return

        self._history.append(text)
        self._history_index = len(self._history)
        inp.value = ""

        self.post_message(self.SendMessage(text))

    def on_key(self, event: "events.Key") -> None:
        """Handle up/down arrow for history navigation."""
        inp = self.query_one("#prompt-input", Input)
        if inp.has_focus:
            return  # Let Input handle its own keys

        if event.key == "up" and self._history:
            self._history_index = max(0, self._history_index - 1)
            inp.value = self._history[self._history_index]
            inp.action_cursor_right()
        elif event.key == "down" and self._history:
            self._history_index = min(len(self._history), self._history_index + 1)
            if self._history_index < len(self._history):
                inp.value = self._history[self._history_index]
            else:
                inp.value = ""
                self._history_index = len(self._history)
