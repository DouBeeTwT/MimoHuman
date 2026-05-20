"""Status bar widget showing agent info."""

from textual.widgets import Static


class AgentStatus(Static):
    """Header widget displaying flagship / speed models and status."""

    def __init__(self) -> None:
        super().__init__("", id="agent-status")

    def update_status(
        self,
        agent_name: str = "",
        pro_label: str = "",
        flash_label: str = "",
        status: str = "idle",
        tokens: int = 0,
    ) -> None:
        """Update the status display."""
        status_icon = {"idle": "○", "thinking": "◉", "executing": "◎", "error": "✕"}.get(
            status, "○"
        )
        parts = []
        if agent_name:
            parts.append(f"[bold]{agent_name}[/bold]")
        if pro_label:
            parts.append(f"Pro: {pro_label}")
        if flash_label:
            parts.append(f"Flash: {flash_label}")
        parts.append(f"{status_icon} {status}")
        if tokens:
            parts.append(f"Tokens: {tokens}")

        self.update(" | ".join(parts))
