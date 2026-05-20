"""OpenAI provider implementation."""

from typing import Any, AsyncGenerator

from mimohuman.core.exceptions import ProviderAPIError, ProviderAuthenticationError, ProviderRateLimitError
from mimohuman.core.message import AssistantMessage, Message, ToolCall
from mimohuman.core.provider import LLMProvider, ProviderConfig, StreamEvent, StreamEventType
from mimohuman.core.tool import Tool


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API (and compatible services)."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        try:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                default_headers=config.extra_headers,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install mimohuman[openai]"
            )

    def supports_vision(self) -> bool:
        return True

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        formatted_messages = [msg.to_provider_format("openai") for msg in messages]

        request_kwargs: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            request_kwargs["tools"] = [t.to_openai_schema() for t in tools]

        try:
            stream = await self._client.chat.completions.create(**request_kwargs)

            current_tool_id: str | None = None
            current_tool_name: str | None = None
            usage_data: dict[str, int] = {}

            async for chunk in stream:
                if chunk.usage:
                    usage_data = {
                        "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    }
                    continue

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        data={"delta": delta.content},
                    )

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.id:
                            current_tool_id = tc_delta.id
                            current_tool_name = tc_delta.function.name if tc_delta.function else ""
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                data={"id": current_tool_id, "name": current_tool_name},
                            )

                        if tc_delta.function and tc_delta.function.arguments:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                data={
                                    "id": current_tool_id,
                                    "name": current_tool_name,
                                    "delta": tc_delta.function.arguments,
                                },
                            )

                if chunk.choices[0].finish_reason:
                    if current_tool_id:
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_END,
                            data={"id": current_tool_id},
                        )
                        current_tool_id = None
                        current_tool_name = None

            yield StreamEvent(type=StreamEventType.DONE, data=usage_data)

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
        formatted_messages = [msg.to_provider_format("openai") for msg in messages]

        request_kwargs: dict[str, Any] = {
            "model": model or self.config.default_model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }

        if tools:
            request_kwargs["tools"] = [t.to_openai_schema() for t in tools]

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            msg = choice.message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    import json

                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_calls.append(
                        ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                    )

            return AssistantMessage(
                content=msg.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
            )

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                raise ProviderAuthenticationError(error_msg) from e
            elif "429" in error_msg:
                raise ProviderRateLimitError(error_msg) from e
            raise ProviderAPIError(error_msg) from e
