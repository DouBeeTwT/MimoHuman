"""Textual App entry point for the MimoHuman TUI."""

import os

from textual.app import App

from mimohuman.core.config import MimoConfig
from mimohuman.core.provider_registry import (
    DEFAULT_PROVIDER,
    get_provider_entry,
    list_providers,
)
from mimohuman.tui.controller import TUIController, _build_agent
from mimohuman.tui.screens.chat_screen import ChatScreen
from mimohuman.tui.screens.help_screen import HelpScreen
from mimohuman.tui.screens.settings_screen import SettingsScreen

APP_CSS = """
Screen {
    layout: vertical;
}

#agent-status {
    dock: top;
    height: 1;
    padding: 0 1;
    background: $panel;
    color: $text-muted;
    text-style: bold;
}

#chat-view {
    height: 1fr;
    padding: 0 1;
    overflow-y: auto;
}

#input-bar {
    dock: bottom;
    height: 3;
    padding: 0 1;
    background: $panel;
    align: center middle;
}

#prompt-input {
    width: 1fr;
    margin-right: 1;
}

#send-button {
    min-width: 8;
}

.chat-message {
    padding: 1 0;
}

.user-message {
    color: $accent;
}

.assistant-message {
    color: $text;
}

.system-message {
    color: $text-muted;
    text-style: italic;
}

.tool-message {
    color: $warning;
    padding-left: 2;
}

.error-message {
    color: $error;
}

#help-title {
    dock: top;
    height: 1;
    padding: 0 1;
    background: $panel;
    text-style: bold;
}

ToolCallDisplay {
    padding: 0 1;
    margin: 1 0;
    border-top: solid yellow;
    border-bottom: solid yellow;
}

ThinkingDisplay {
    padding: 0 1;
    margin: 1 0;
    border-top: solid grey;
    border-bottom: solid grey;
}
"""


class MimoHumanApp(App):
    """MimoHuman -- Generic AI Agent Framework with TUI."""

    CSS = APP_CSS
    TITLE = "MimoHuman"
    SUB_TITLE = "AI Agent Framework"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "push_screen('settings')", "Settings"),
        ("ctrl+h", "push_screen('help')", "Help"),
    ]

    def __init__(self, controller: TUIController | None = None) -> None:
        super().__init__()
        self._controller = controller

    def on_mount(self) -> None:
        """Register screens and push the main chat screen."""
        self.install_screen(SettingsScreen(), name="settings")
        self.install_screen(HelpScreen(), name="help")
        if self._controller is None:
            self._controller = _build_default_controller()
        self.push_screen(ChatScreen(self._controller))

    @property
    def controller(self) -> TUIController:
        if self._controller is None:
            self._controller = _build_default_controller()
        return self._controller


def _build_default_controller() -> TUIController:
    """Build a TUIController using provider registry defaults.

    Respects MIMOHUMAN_PROVIDER / MIMOHUMAN_MODEL env vars.
    Saved config takes priority over env vars.
    """
    config = MimoConfig.load()

    pro_key = os.environ.get("MIMOHUMAN_PROVIDER", config.pro_provider or DEFAULT_PROVIDER)
    pro_model = os.environ.get("MIMOHUMAN_MODEL", config.pro_model or "mimo-pro")

    pro_entry = get_provider_entry(pro_key)
    if pro_entry is None:
        available = [e.key for e in list_providers()]
        raise ValueError(f"Unknown provider '{pro_key}'. Available: {', '.join(available)}")

    flash_key = config.flash_provider or pro_key
    flash_model = config.flash_model or "mimo-fast"

    agent = _build_agent(
        pro_entry,
        pro_model,
        base_url_override=config.get_url(pro_key) or None,
        api_key_override=config.get_key(pro_key) or None,
    )
    return TUIController(
        agent,
        config=config,
        pro_provider=pro_key,
        pro_model=pro_model,
        flash_provider=flash_key,
        flash_model=flash_model,
    )


def main() -> None:
    """Entry point for the `mimohuman` console script."""
    controller = _build_default_controller()
    app = MimoHumanApp(controller)
    app.run()


if __name__ == "__main__":
    main()
