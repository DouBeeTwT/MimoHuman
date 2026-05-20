"""Settings overlay — configure providers and assign Pro / Flash models."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    Select,
    Static,
)

from mimohuman.core.provider_registry import list_providers


class SettingsScreen(ModalScreen[None]):
    """Modal overlay for provider configuration and model assignment."""

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
        ("ctrl+s", "apply", "Apply"),
    ]

    CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-dialog {
        width: 82;
        height: auto;
        max-height: 52;
        background: $surface;
        border: thick blue;
        padding: 1 2;
        overflow-y: auto;
    }

    #settings-title {
        text-style: bold;
        text-align: center;
        padding-bottom: 1;
        border-bottom: dashed grey;
        width: 100%;
        margin-bottom: 1;
    }

    .section-header {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 1;
    }

    #provider-select {
        width: 1fr;
        margin-bottom: 1;
    }

    .field-row {
        height: auto;
        margin-bottom: 1;
    }

    .field-label {
        color: $text-muted;
        width: 10;
    }

    .field-input {
        width: 1fr;
    }

    #assignment-area {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }

    #assignment-area Label {
        height: 1;
        width: 10;
    }

    #assignment-area Select {
        width: 20;
    }

    #assignment-area Input {
        width: 1fr;
    }

    #settings-buttons {
        margin-top: 2;
        margin-bottom: 1;
        align: center middle;
        width: 100%;
    }

    #settings-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._providers = list_providers()
        self._url_values: dict[str, str] = {}
        self._key_values: dict[str, str] = {}
        self._current_provider: str = ""

    def compose(self) -> ComposeResult:
        from mimohuman.tui.app import MimoHumanApp
        app: MimoHumanApp = self.app  # type: ignore[assignment]
        ctrl = app.controller

        for entry in self._providers:
            self._url_values[entry.key] = ctrl.get_url(entry.key)
            self._key_values[entry.key] = ctrl.get_key(entry.key)

        provider_opts = [(entry.name, entry.key) for entry in self._providers]

        with Container(id="settings-dialog"):
            yield Static("⚙ Settings", id="settings-title")

            yield Label("Provider Configuration", classes="section-header")

            self._current_provider = ctrl.pro_provider
            yield Select(provider_opts, value=ctrl.pro_provider, id="provider-select")

            with Horizontal(classes="field-row"):
                yield Label("URL", classes="field-label")
                yield Input(
                    value=self._url_values.get(ctrl.pro_provider, ""),
                    placeholder="https://api.example.com/v1",
                    id="url-input",
                    classes="field-input",
                )
            with Horizontal(classes="field-row"):
                yield Label("API Key", classes="field-label")
                yield Input(
                    value=self._key_values.get(ctrl.pro_provider, ""),
                    placeholder="sk-...",
                    password=True,
                    id="key-input",
                    classes="field-input",
                )

            yield Label("Model Assignment  (* required)", classes="section-header")
            with Container(id="assignment-area"):
                yield Label("Pro")
                with Horizontal(classes="field-row"):
                    yield Select(
                        provider_opts,
                        value=ctrl.pro_provider,
                        id="pro-provider",
                    )
                    yield Input(
                        value=ctrl.pro_model,
                        placeholder="Model name, e.g. mimo-pro",
                        id="pro-model",
                        classes="field-input",
                    )

                yield Label("Flash")
                with Horizontal(classes="field-row"):
                    yield Select(
                        provider_opts,
                        value=ctrl.flash_provider,
                        id="flash-provider",
                    )
                    yield Input(
                        value=ctrl.flash_model,
                        placeholder="Model name, e.g. mimo-fast",
                        id="flash-model",
                        classes="field-input",
                    )

            with Horizontal(id="settings-buttons"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Apply", variant="primary", id="btn-apply")

    def on_mount(self) -> None:
        try:
            self.query_one("#provider-select", Select).focus()
        except Exception:
            pass

    @on(Select.Changed, "#provider-select")
    def _on_provider_changed(self, event: Select.Changed) -> None:
        if self._current_provider:
            try:
                url_input = self.query_one("#url-input", Input)
                key_input = self.query_one("#key-input", Input)
                self._url_values[self._current_provider] = url_input.value
                self._key_values[self._current_provider] = key_input.value
            except Exception:
                pass

        new_key: str = event.value  # type: ignore[assignment]
        if not new_key:
            return

        self._current_provider = new_key

        try:
            url_input = self.query_one("#url-input", Input)
            key_input = self.query_one("#key-input", Input)
            url_input.value = self._url_values.get(new_key, "")
            key_input.value = self._key_values.get(new_key, "")
        except Exception:
            pass

    @on(Button.Pressed, "#btn-cancel")
    def _on_cancel(self) -> None:
        self.dismiss()

    def action_apply(self) -> None:
        self._do_apply()

    @on(Button.Pressed, "#btn-apply")
    def _on_apply(self) -> None:
        self._do_apply()

    def _do_apply(self) -> None:
        from mimohuman.tui.app import MimoHumanApp
        app: MimoHumanApp = self.app  # type: ignore[assignment]
        ctrl = app.controller

        # Flush current URL/Key before saving
        provider_sel = self.query_one("#provider-select", Select)
        current_provider: str = provider_sel.value  # type: ignore[assignment]
        url_input = self.query_one("#url-input", Input)
        key_input = self.query_one("#key-input", Input)
        self._url_values[current_provider] = url_input.value
        self._key_values[current_provider] = key_input.value

        # Read Pro / Flash assignments
        pro_provider_sel = self.query_one("#pro-provider", Select)
        pro_model_input = self.query_one("#pro-model", Input)
        flash_provider_sel = self.query_one("#flash-provider", Select)
        flash_model_input = self.query_one("#flash-model", Input)

        pro_provider: str = pro_provider_sel.value  # type: ignore[assignment]
        pro_model = pro_model_input.value.strip()
        flash_provider: str = flash_provider_sel.value  # type: ignore[assignment]
        flash_model = flash_model_input.value.strip()

        if not pro_model or not flash_model:
            self.notify("Both Pro and Flash model names must be entered.", severity="error")
            return

        if pro_provider == flash_provider and pro_model == flash_model:
            self.notify("Pro and Flash must be different models.", severity="error")
            return

        # Persist per-provider URL / key overrides
        for entry in self._providers:
            ctrl.set_provider_config(
                entry.key,
                self._url_values.get(entry.key, ""),
                self._key_values.get(entry.key, ""),
            )

        # Apply model assignments
        try:
            ctrl.set_model_assignments(
                pro_provider, pro_model, flash_provider, flash_model
            )
        except Exception:
            self.notify("Failed to apply settings.", severity="error")
            return

        self.dismiss()
