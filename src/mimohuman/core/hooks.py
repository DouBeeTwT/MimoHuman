"""Lifecycle hook system for injecting custom behavior."""

from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class HookPoint(str, Enum):
    """Points in the agent lifecycle where hooks can fire."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_STREAM_EVENT = "on_stream_event"
    ON_ERROR = "on_error"


HookHandler = Callable[["HookContext"], Awaitable[None]]


class HookContext(BaseModel):
    """Context passed to each hook handler.

    Handlers can inspect and modify the data dict, or set cancel=True
    to abort the current action.
    """

    hook_point: HookPoint
    agent_name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    cancel: bool = False

    model_config = {"arbitrary_types_allowed": True}


class HookManager:
    """Manages registered hooks and fires them at lifecycle points.

    Handlers are called in priority order (higher priority first).
    If any handler sets context.cancel=True, subsequent handlers for
    that hook point are skipped, and the triggering action is aborted.
    """

    def __init__(self) -> None:
        self._handlers: dict[HookPoint, list[tuple[int, HookHandler]]] = {
            p: [] for p in HookPoint
        }

    def register(
        self,
        point: HookPoint,
        handler: HookHandler,
        priority: int = 0,
    ) -> None:
        """Register a handler for a hook point."""
        self._handlers[point].append((priority, handler))
        self._handlers[point].sort(key=lambda x: x[0], reverse=True)

    def unregister(self, point: HookPoint, handler: HookHandler) -> None:
        """Remove a handler from a hook point."""
        self._handlers[point] = [
            (p, h) for p, h in self._handlers[point] if h is not handler
        ]

    async def fire(self, context: HookContext) -> HookContext:
        """Fire all handlers for the given hook point.

        Returns the (possibly modified) context. If any handler sets
        cancel=True, stops and returns immediately.
        """
        for _priority, handler in self._handlers[context.hook_point]:
            await handler(context)
            if context.cancel:
                break
        return context
