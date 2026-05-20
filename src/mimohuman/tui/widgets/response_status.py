"""Response status bar fixed below the input area."""

from textual.widgets import Static


class ResponseStatusBar(Static):
    """Fixed bar showing Speed, Ctx, and Cfs metrics.

    Positioned between the chat view and the input bar. Cfs starts as "..."
    while the Flash model computes the confusion score in the background.
    """

    def __init__(
        self,
        speed_tokens_per_sec: float = 0.0,
        ctx_tokens: int = 0,
        cfs: str = "...",
        **kwargs,
    ) -> None:
        self._speed = speed_tokens_per_sec
        self._ctx = ctx_tokens
        self._cfs = cfs
        super().__init__(self._format_text(), **kwargs)

    def _format_text(self) -> str:
        ctx_k = self._ctx / 1000
        return (
            f"Speed: {self._speed:.1f} Tokens/s | "
            f"Ctx: {ctx_k:.1f}k | "
            f"Cfs: {self._cfs}"
        )

    def update_metrics(
        self,
        speed: float | None = None,
        ctx_tokens: int | None = None,
        cfs: str | None = None,
    ) -> None:
        """Update displayed metrics. Only non-None values are changed."""
        if speed is not None:
            self._speed = speed
        if ctx_tokens is not None:
            self._ctx = ctx_tokens
        if cfs is not None:
            self._cfs = cfs
        self.update(self._format_text())

    def set_cfs(self, value: float | None) -> None:
        """Update the confusion score from a float percentage."""
        if value is not None:
            self._cfs = f"{value:.1f}%"
        else:
            self._cfs = "--"
        self.update(self._format_text())

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._speed = 0.0
        self._ctx = 0
        self._cfs = "..."
        self.update(self._format_text())
