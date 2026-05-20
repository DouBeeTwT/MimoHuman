"""Tests for hook system."""

import pytest

from mimohuman.core.hooks import HookContext, HookManager, HookPoint


@pytest.mark.asyncio
async def test_hook_fires_handler() -> None:
    hm = HookManager()
    called = []

    async def my_handler(ctx: HookContext) -> None:
        called.append(ctx.hook_point)

    hm.register(HookPoint.AGENT_START, my_handler)

    ctx = HookContext(hook_point=HookPoint.AGENT_START, agent_name="test")
    await hm.fire(ctx)

    assert len(called) == 1
    assert called[0] == HookPoint.AGENT_START


@pytest.mark.asyncio
async def test_hook_cancel_stops_chain() -> None:
    hm = HookManager()
    first_called = False
    second_called = False

    async def first_handler(ctx: HookContext) -> None:
        nonlocal first_called
        first_called = True
        ctx.cancel = True

    async def second_handler(ctx: HookContext) -> None:
        nonlocal second_called
        second_called = True

    hm.register(HookPoint.BEFORE_LLM_CALL, first_handler, priority=10)
    hm.register(HookPoint.BEFORE_LLM_CALL, second_handler, priority=5)

    ctx = HookContext(hook_point=HookPoint.BEFORE_LLM_CALL, agent_name="test")
    result = await hm.fire(ctx)

    assert first_called is True
    assert second_called is False
    assert result.cancel is True


@pytest.mark.asyncio
async def test_hook_unregister() -> None:
    hm = HookManager()
    called = False

    async def my_handler(ctx: HookContext) -> None:
        nonlocal called
        called = True

    hm.register(HookPoint.AGENT_END, my_handler)
    hm.unregister(HookPoint.AGENT_END, my_handler)

    ctx = HookContext(hook_point=HookPoint.AGENT_END, agent_name="test")
    await hm.fire(ctx)

    assert called is False
