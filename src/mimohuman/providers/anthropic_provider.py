"""Anthropic provider implementation."""

from typing import Any, AsyncGenerator

from mimohuman.core.exceptions import ProviderAPIError, ProviderAuthenticationError, ProviderRateLimitError
from mimohuman.core.message import AssistantMessage, Message
from mimohuman.core.provider import LLMProvider, ProviderConfig, StreamEvent, StreamEventType
from mimohuman.core.tool import Tool


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=config.api_key,
                base_url=config.base_url,
                default_headers=config.extra_headers,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install mimohuman[anthropic]"
            )

    def supports_thinking(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        system_prompt = ""
        formatted_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role.value == "system":
                system_prompt = msg.content
            else:
                formatted_messages.append(msg.to_provider_format("anthropic"))

        request_kwargs: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = [t.to_anthropic_schema() for t in tools]

        try:
            async with self._client.messages.stream(**request_kwargs) as stream:
                current_tool_id: str | None = None
                current_tool_name: str | None = None

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                data={"id": block.id, "name": block.name},
                            )

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                data={"delta": delta.text},
                            )
                        elif delta.type == "thinking_delta":
                            yield StreamEvent(
                                type=StreamEventType.THINKING_DELTA,
                                data={"delta": delta.thinking},
                            )
                        elif delta.type == "input_json_delta":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                data={
                                    "id": current_tool_id,
                                    "name": current_tool_name,
                                    "delta": delta.partial_json,
                                },
                            )

                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_END,
                                data={"id": current_tool_id},
                            )
                            current_tool_id = None
                            current_tool_name = None

                    elif event.type == "message_stop":
                        yield StreamEvent(type=StreamEventType.DONE, data={})

                    elif event.type == "error":
                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            data={"message": str(event.error)},
                        )

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                raise ProviderAuthenticationError(error_msg) from e
            elif "429" in error_msg:
                raise ProviderRateLimitError(error_msg) from e
            raise ProviderAPIError(error_msg) from e

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        system_prompt = ""
        formatted_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role.value == "system":
                system_prompt = msg.content
            else:
                formatted_messages.append(msg.to_provider_format("anthropic"))

        request_kwargs: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = [t.to_anthropic_schema() for t in tools]

        try:
            response = await self._client.messages.create(**request_kwargs)

            content_text = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    from mimohuman.core.message import ToolCall

                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input)
                    )

            return AssistantMessage(
                content=content_text,
                tool_calls=tool_calls,
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                raise ProviderAuthenticationError(error_msg) from e
            elif "429" in error_msg:
                raise ProviderRateLimitError(error_msg) from e
            raise ProviderAPIError(error_msg) from e
