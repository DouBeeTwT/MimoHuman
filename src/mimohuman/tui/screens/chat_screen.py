"""Main chat interaction screen."""

from textual import on
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer

from mimohuman.core.provider import StreamEventType
from mimohuman.tui.controller import TUIController
from mimohuman.tui.widgets.agent_status import AgentStatus
from mimohuman.tui.widgets.chat_view import ChatView
from mimohuman.tui.widgets.input_bar import InputBar


class ChatScreen(Screen):
    """The main screen where users chat with the agent."""

    BINDINGS = [
        ("escape", "focus_input", "Focus Input"),
        ("ctrl+s", "push_screen('settings')", "Settings"),
    ]

    def __init__(self, controller: TUIController) -> None:
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        yield AgentStatus()
        yield ChatView()
        yield InputBar()
        yield Footer()

    def on_mount(self) -> None:
        """Set up initial state when the screen is mounted."""
        self._refresh_status()

    def _on_screen_resume(self) -> None:
        """Refresh status when this screen becomes active again."""
        self._refresh_status()

    def _refresh_status(self) -> None:
        """Sync the status bar with current controller state."""
        status = self.query_one(AgentStatus)
        status.update_status(
            agent_name=self.controller.agent.config.name,
            pro_label=self.controller.pro_label,
            flash_label=self.controller.flash_label,
            status="idle",
        )

    @on(InputBar.SendMessage)
    def on_send_message(self, event: InputBar.SendMessage) -> None:
        """Handle a user message submission."""
        text = event.text.strip()
        if not text:
            return

        # Handle slash commands synchronously
        if text.startswith("/"):
            self._handle_command(text)
            return

        chat_view = self.query_one(ChatView)
        status = self.query_one(AgentStatus)

        # Display user message immediately
        chat_view.add_user_message(text)
        status.update_status(
            agent_name=self.controller.agent.config.name,
            pro_label=self.controller.pro_label,
            flash_label=self.controller.flash_label,
            status="thinking",
        )

        # Run streaming in a worker so the UI stays responsive
        self.run_worker(self._stream_response(text), exclusive=True)

    async def _stream_response(self, text: str) -> None:
        """Stream the agent response in a background worker."""
        chat_view = self.query_one(ChatView)
        status = self.query_one(AgentStatus)

        current_response = ""
        stream_widget = None
        try:
            async for stream_event in self.controller.send_message(text):
                match stream_event.type:
                    case StreamEventType.TEXT_DELTA:
                        current_response += stream_event.data.get("delta", "")
                        if stream_widget is None:
                            stream_widget = chat_view.begin_streaming()
                        stream_widget.update(current_response)
                    case StreamEventType.TOOL_CALL_START:
                        tc_name = stream_event.data.get("name", "unknown")
                        chat_view.add_system_message(f"Calling tool: {tc_name}...")
                        status.update_status(
                            agent_name=self.controller.agent.config.name,
                            pro_label=self.controller.pro_label,
                            flash_label=self.controller.flash_label,
                            status="executing",
                        )
                    case StreamEventType.TOOL_RESULT:
                        tc_name = stream_event.data.get("name", "unknown")
                        result = stream_event.data.get("result", "")
                        chat_view.add_tool_message(tc_name, "", str(result)[:200])
                    case StreamEventType.ERROR:
                        chat_view.add_error(stream_event.data.get("message", "Unknown error"))
                    case StreamEventType.AGENT_END:
                        if stream_widget is None and current_response:
                            chat_view.add_assistant_message(current_response)

        except Exception as e:
            chat_view.add_error(str(e))

        finally:
            status.update_status(
                agent_name=self.controller.agent.config.name,
                pro_label=self.controller.pro_label,
                flash_label=self.controller.flash_label,
                status="idle",
            )

    def _handle_command(self, text: str) -> None:
        """Handle slash commands."""
        chat_view = self.query_one(ChatView)

        if text == "/clear":
            self.controller.clear_conversation()
            chat_view.add_system_message("Conversation cleared.")
        elif text == "/help":
            self.app.push_screen("help")
        elif text == "/settings":
            self.app.push_screen("settings")
        elif text == "/tools":
            tools = self.controller.tool_registry.list_tools()
            if tools:
                tool_list = "\n".join(
                    f"  - {t.name}: {t.description}" for t in tools
                )
                chat_view.add_system_message(f"Available tools:\n{tool_list}")
            else:
                chat_view.add_system_message("No tools registered.")
        elif text.startswith("/model "):
            chat_view.add_system_message(
                "Use /settings or Ctrl+S to switch provider and model."
            )
        else:
            chat_view.add_system_message(f"Unknown command: {text}")

    def action_focus_input(self) -> None:
        """Focus the input bar."""
        try:
            inp = self.query_one("#prompt-input")
            inp.focus()
        except Exception:
            pass
