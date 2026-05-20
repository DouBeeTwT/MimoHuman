"""Bridge between the TUI and the Agent core.

This is the ONLY module in mimohuman.tui that imports from mimohuman.core.
Widgets and screens receive data through this controller or via Textual messages.
"""

import os
from pathlib import Path

from mimohuman.core.agent import Agent, AgentConfig
from mimohuman.core.config import MimoConfig
from mimohuman.core.conversation import Conversation
from mimohuman.core.provider import LLMProvider, ProviderConfig
from mimohuman.core.provider_registry import (
    DEFAULT_MODEL_TIER,
    DEFAULT_PROVIDER,
    ProviderEntry,
    get_provider_entry,
)
from mimohuman.core.tool import ToolRegistry


def _load_soul_prompt() -> str:
    """Load the Pro model system prompt from prompts/soul.md."""
    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "soul.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are a helpful AI assistant."


def _resolve_api_key(entry: ProviderEntry, override: str | None = None) -> str:
    if override is not None and override.strip():
        return override.strip()
    if not entry.api_key_env:
        return "not-configured"
    key = os.environ.get(entry.api_key_env, "")
    return key if key else "not-configured"


def _build_agent(
    entry: ProviderEntry,
    model_name: str,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Agent:
    api_key = _resolve_api_key(entry, api_key_override)
    base_url = base_url_override if base_url_override else entry.base_url

    config = ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        default_model=model_name,
        extra_headers=entry.extra_headers,
    )
    if entry.provider_type == "anthropic":
        from mimohuman.providers.anthropic_provider import AnthropicProvider
        provider: LLMProvider = AnthropicProvider(config)
    elif entry.provider_type == "openai":
        from mimohuman.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {entry.provider_type}")

    agent_config = AgentConfig(
        name="MimoHuman",
        system_prompt=_load_soul_prompt(),
        model=model_name,
    )
    return Agent(
        config=agent_config,
        provider=provider,
        tool_registry=tool_registry or ToolRegistry(),
    )


class TUIController:
    """Owns the Agent lifecycle and translates stream events for the TUI.

    Maintains per-provider connection overrides and two independent model
    assignments (Pro + Flash) for task-based routing.
    """

    def __init__(
        self,
        agent: Agent,
        config: MimoConfig | None = None,
        pro_provider: str = DEFAULT_PROVIDER,
        pro_model: str = "mimo-pro",
        flash_provider: str | None = None,
        flash_model: str = "mimo-fast",
    ) -> None:
        self._agent = agent
        self._flash_agent: Agent | None = None
        self._conversation = Conversation()
        self._config = config or MimoConfig()

        self._pro_provider = self._config.pro_provider or pro_provider
        self._pro_model = self._config.pro_model or pro_model
        self._flash_provider = self._config.flash_provider or flash_provider or pro_provider
        self._flash_model = self._config.flash_model or flash_model

        self._build_flash_agent()

    # ── provider override accessors ──────────────────────────────

    def get_url(self, provider_key: str) -> str:
        cfg_val = self._config.get_url(provider_key)
        if cfg_val:
            return cfg_val
        entry = get_provider_entry(provider_key)
        return entry.base_url or "" if entry else ""

    def get_key(self, provider_key: str) -> str:
        cfg_val = self._config.get_key(provider_key)
        if cfg_val:
            return cfg_val
        entry = get_provider_entry(provider_key)
        if entry and entry.api_key_env:
            return os.environ.get(entry.api_key_env, "")
        return ""

    def set_provider_config(
        self, provider_key: str, url: str, api_key: str
    ) -> None:
        self._config.set_provider(provider_key, url.strip(), api_key.strip())
        self._config.save()

    # ── model assignment accessors ───────────────────────────────

    @property
    def pro_provider(self) -> str:
        return self._pro_provider

    @property
    def pro_model(self) -> str:
        return self._pro_model

    @property
    def flash_provider(self) -> str:
        return self._flash_provider

    @property
    def flash_model(self) -> str:
        return self._flash_model

    def set_model_assignments(
        self,
        pro_provider: str,
        pro_model: str,
        flash_provider: str,
        flash_model: str,
    ) -> None:
        self._pro_provider = pro_provider
        self._pro_model = pro_model
        self._flash_provider = flash_provider
        self._flash_model = flash_model

        self._config.pro_provider = pro_provider
        self._config.pro_model = pro_model
        self._config.flash_provider = flash_provider
        self._config.flash_model = flash_model
        self._config.save()

        self._rebuild_agent(self._pro_provider, self._pro_model)
        self._build_flash_agent()

    def _rebuild_agent(self, provider_key: str, model_name: str) -> None:
        entry = get_provider_entry(provider_key)
        if entry is None:
            return
        self._agent = _build_agent(
            entry,
            model_name,
            base_url_override=self._config.get_url(provider_key) or None,
            api_key_override=self._config.get_key(provider_key) or None,
            tool_registry=self._agent.tool_registry,
        )

    def _build_flash_agent(self) -> None:
        entry = get_provider_entry(self._flash_provider)
        if entry is None:
            self._flash_agent = None
            return
        self._flash_agent = _build_agent(
            entry,
            self._flash_model,
            base_url_override=self._config.get_url(self._flash_provider) or None,
            api_key_override=self._config.get_key(self._flash_provider) or None,
            tool_registry=self._agent.tool_registry if self._agent else ToolRegistry(),
        )

    # ── display helpers ──────────────────────────────────────────

    @property
    def pro_label(self) -> str:
        entry = get_provider_entry(self._pro_provider)
        provider_name = entry.name if entry else self._pro_provider
        return f"{provider_name} > {self._pro_model}"

    @property
    def flash_label(self) -> str:
        entry = get_provider_entry(self._flash_provider)
        provider_name = entry.name if entry else self._flash_provider
        return f"{provider_name} > {self._flash_model}"

    @property
    def current_model_label(self) -> str:
        return self.pro_label

    @property
    def current_model_name(self) -> str:
        return self._pro_model

    @property
    def provider_key(self) -> str:
        return self._pro_provider

    @property
    def provider_display_name(self) -> str:
        entry = get_provider_entry(self._pro_provider)
        return entry.name if entry else self._pro_provider

    # ── chat flow ────────────────────────────────────────────────

    async def send_message(self, text: str):
        async for event in self._agent.run(text, self._conversation):
            yield event

    def get_conversation(self) -> Conversation:
        return self._conversation

    def clear_conversation(self) -> None:
        self._conversation.clear()

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def flash_agent(self) -> Agent | None:
        return self._flash_agent

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._agent.tool_registry
