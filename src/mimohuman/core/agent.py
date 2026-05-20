"""Agent -- the central orchestration unit.

Ties together a provider, tool registry, and conversation to implement
the canonical agent loop: call LLM -> execute tools -> repeat.
"""

import json
import time
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

from mimohuman.core.conversation import Conversation
from mimohuman.core.exceptions import MaxToolRoundsExceeded
from mimohuman.core.hooks import HookContext, HookManager, HookPoint
from mimohuman.core.message import (
    AssistantMessage,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from mimohuman.core.provider import LLMProvider, StreamEvent, StreamEventType
from mimohuman.core.tool import ToolRegistry


class AgentConfig(BaseModel):
    """Configuration for an Agent."""

    name: str = "Agent"
    system_prompt: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    max_tool_rounds: int = 10
    stop_sequences: list[str] = Field(default_factory=list)


class Agent:
    """A single AI agent with tool-calling capability.

    Ties a provider, tool registry, and conversation together and
    implements the standard agent loop: call LLM -> parse tool calls ->
    execute tools -> feed results back -> repeat.
    """

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        hook_manager: HookManager | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tool_registry = tool_registry or ToolRegistry()
        self.hook_manager = hook_manager or HookManager()

    async def run(
        self,
        user_input: str,
        conversation: Conversation | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the agent loop for a user input.

        Yields StreamEvent objects at every stage so callers (TUI, API,
        orchestrator) can observe progress in real time.
        """
        conv = conversation or Conversation()
        start_time = time.monotonic()
        total_input_tokens = 0
        total_output_tokens = 0

        # AGENT_START hook
        ctx = HookContext(
            hook_point=HookPoint.AGENT_START,
            agent_name=self.config.name,
            data={"user_input": user_input},
        )
        await self.hook_manager.fire(ctx)
        if ctx.cancel:
            return

        yield StreamEvent(type=StreamEventType.AGENT_START, data={"agent": self.config.name})

        # Add user message
        user_msg = UserMessage(content=user_input)
        conv.add(user_msg)

        # Build the full message list
        messages = self._build_message_list(conv)
        tools = self.tool_registry.list_tools()

        round_count = 0
        while round_count < self.config.max_tool_rounds:
            round_count += 1
            if round_count > 1:
                yield StreamEvent(
                    type=StreamEventType.ROUND_START,
                    data={"round": round_count},
                )

            # BEFORE_LLM_CALL hook
            llm_ctx = HookContext(
                hook_point=HookPoint.BEFORE_LLM_CALL,
                agent_name=self.config.name,
                data={"messages": messages, "tools": tools, "round": round_count},
            )
            await self.hook_manager.fire(llm_ctx)
            if llm_ctx.cancel:
                break

            # Collect the assistant response from the stream
            content_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            pending_tool_calls: dict[str, dict[str, Any]] = {}

            async for event in self.provider.stream(
                messages=messages,
                tools=tools if tools else None,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ):
                # ON_STREAM_EVENT hook
                event_ctx = HookContext(
                    hook_point=HookPoint.ON_STREAM_EVENT,
                    agent_name=self.config.name,
                    data={"event": event},
                )
                await self.hook_manager.fire(event_ctx)
                if event_ctx.cancel:
                    break

                if event.type == StreamEventType.TEXT_DELTA:
                    content_parts.append(event.data.get("delta", ""))

                elif event.type == StreamEventType.TOOL_CALL_START:
                    tc_id = event.data["id"]
                    tc_name = event.data["name"]
                    pending_tool_calls[tc_id] = {"id": tc_id, "name": tc_name, "arguments": ""}

                elif event.type == StreamEventType.TOOL_CALL_DELTA:
                    tc_id = event.data.get("id", "")
                    if tc_id in pending_tool_calls:
                        pending_tool_calls[tc_id]["arguments"] += event.data.get("delta", "")

                elif event.type == StreamEventType.TOOL_CALL_END:
                    tc_id = event.data["id"]
                    if tc_id in pending_tool_calls:
                        tc = pending_tool_calls.pop(tc_id)
                        try:
                            args = json.loads(tc["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        tool_calls.append(
                            ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                        )

                yield event

                if event.type == StreamEventType.DONE:
                    total_input_tokens += event.data.get("input_tokens", 0)
                    total_output_tokens += event.data.get("output_tokens", 0)

            # Build assistant message
            content = "".join(content_parts)
            assistant_msg = AssistantMessage(
                content=content,
                tool_calls=tool_calls,
            )
            conv.add(assistant_msg)

            # AFTER_LLM_CALL hook
            after_ctx = HookContext(
                hook_point=HookPoint.AFTER_LLM_CALL,
                agent_name=self.config.name,
                data={"assistant_message": assistant_msg},
            )
            await self.hook_manager.fire(after_ctx)

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tools
            for tc in tool_calls:
                # BEFORE_TOOL_CALL hook
                tool_ctx = HookContext(
                    hook_point=HookPoint.BEFORE_TOOL_CALL,
                    agent_name=self.config.name,
                    data={"tool_call": tc},
                )
                await self.hook_manager.fire(tool_ctx)
                if tool_ctx.cancel:
                    continue

                try:
                    result = await self.tool_registry.execute(tc.name, **tc.arguments)
                    is_error = False
                    result_text = str(result)
                except Exception as e:
                    result_text = str(e)
                    is_error = True

                tool_msg = ToolResultMessage(
                    content=result_text,
                    tool_call_id=tc.id,
                    name=tc.name,
                    is_error=is_error,
                )
                conv.add(tool_msg)

                yield StreamEvent(
                    type=StreamEventType.TOOL_RESULT,
                    data={
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "result": result_text,
                        "is_error": is_error,
                    },
                )

                # AFTER_TOOL_CALL hook
                after_tool_ctx = HookContext(
                    hook_point=HookPoint.AFTER_TOOL_CALL,
                    agent_name=self.config.name,
                    data={"tool_call": tc, "result": result_text, "is_error": is_error},
                )
                await self.hook_manager.fire(after_tool_ctx)

            # Rebuild messages for next round
            messages = self._build_message_list(conv)

            if round_count > 1:
                yield StreamEvent(
                    type=StreamEventType.ROUND_END,
                    data={"round": round_count},
                )

        if round_count >= self.config.max_tool_rounds and tool_calls:
            raise MaxToolRoundsExceeded(
                f"Agent '{self.config.name}' exceeded max tool rounds "
                f"({self.config.max_tool_rounds})"
            )

        duration_ms = (time.monotonic() - start_time) * 1000
        yield StreamEvent(
            type=StreamEventType.AGENT_END,
            data={
                "agent": self.config.name,
                "rounds": round_count,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "duration_ms": duration_ms,
            },
        )

        # AGENT_END hook
        end_ctx = HookContext(
            hook_point=HookPoint.AGENT_END,
            agent_name=self.config.name,
            data={"rounds": round_count},
        )
        await self.hook_manager.fire(end_ctx)

    def _build_message_list(self, conversation: Conversation) -> list[Any]:
        """Build the full message list including system prompt."""
        messages: list[Any] = []
        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))
        messages.extend(conversation.get_messages())
        return messages
