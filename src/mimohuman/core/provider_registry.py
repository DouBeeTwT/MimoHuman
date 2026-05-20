"""Provider registry defining available LLM providers and their models.

Each provider offers two model types:
- pro: best quality, highest capability
- flash: faster, lower cost
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelEntry:
    """A single model within a provider."""

    name: str
    label: str


@dataclass
class ProviderEntry:
    """Metadata for a registered LLM provider."""

    key: str
    name: str
    provider_type: Literal["anthropic", "openai"]
    base_url: str | None = None
    api_key_env: str = ""
    models: dict[str, ModelEntry] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDER_REGISTRY: dict[str, ProviderEntry] = {
    "mimo": ProviderEntry(
        key="mimo",
        name="小米 MIMO",
        provider_type="openai",
        base_url="https://api.xiaomimimo.com/v1",
        api_key_env="MIMO_API_KEY",
        models={
            "pro": ModelEntry(
                name="mimo-pro",
                label="MIMO Pro",
            ),
            "flash": ModelEntry(
                name="mimo-fast",
                label="MIMO Flash",
            ),
        },
    ),
    "anthropic": ProviderEntry(
        key="anthropic",
        name="Anthropic",
        provider_type="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        models={
            "pro": ModelEntry(
                name="claude-opus-4-7",
                label="Claude Opus 4.7",
            ),
            "flash": ModelEntry(
                name="claude-sonnet-4-6",
                label="Claude Sonnet 4.6",
            ),
        },
    ),
    "openai": ProviderEntry(
        key="openai",
        name="OpenAI",
        provider_type="openai",
        api_key_env="OPENAI_API_KEY",
        models={
            "pro": ModelEntry(
                name="gpt-5",
                label="GPT-5",
            ),
            "flash": ModelEntry(
                name="gpt-4o",
                label="GPT-4o",
            ),
        },
    ),
}

DEFAULT_PROVIDER = "mimo"
DEFAULT_MODEL_TIER = "pro"


def get_provider_entry(key: str) -> ProviderEntry | None:
    return PROVIDER_REGISTRY.get(key)


def list_providers() -> list[ProviderEntry]:
    return list(PROVIDER_REGISTRY.values())
