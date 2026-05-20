"""Help screen showing keybindings."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Markdown, Static


class HelpScreen(Screen):
    """Modal help screen with keybinding reference."""

    BINDINGS = [("escape", "dismiss", "Close")]

    HELP_TEXT = """\
# MimoHuman Help

## Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+S` | Open settings (switch provider/model) |
| `Ctrl+H` | Toggle this help screen |
| `Escape` | Focus input / Close help |
| `Enter` | Send message (when input focused) |
| `Up/Down` | Navigate input history |

## Commands

Type these in the input bar:

- `/clear` -- Clear conversation history
- `/settings` -- Open settings screen
- `/tools` -- List available tools
- `/help` -- Show this help screen

## Settings

Press `Ctrl+S` or type `/settings` to switch between providers
(小米 MIMO, Anthropic, OpenAI) and choose flagship or speed model.
"""

    def compose(self) -> ComposeResult:
        yield Static("Help", id="help-title")
        yield Markdown(self.HELP_TEXT)
        yield Footer()
