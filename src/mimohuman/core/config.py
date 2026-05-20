"""Persistent configuration for MimoHuman TUI."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg:
        return Path(xdg) / "mimohuman"
    return Path.home() / ".config" / "mimohuman"


def _config_path() -> Path:
    return _config_dir() / "config.json"


@dataclass
class MimoConfig:
    """Persistent config loaded from / saved to ~/.config/mimohuman/config.json.

    Note: API keys are stored in plaintext. Keep the file permissions restricted.
    """

    pro_provider: str = "mimo"
    pro_model: str = "mimo-pro"
    flash_provider: str = "mimo"
    flash_model: str = "mimo-fast"
    providers: dict[str, dict[str, str]] = field(default_factory=dict)

    def get_url(self, provider_key: str) -> str:
        return self.providers.get(provider_key, {}).get("url", "")

    def get_key(self, provider_key: str) -> str:
        return self.providers.get(provider_key, {}).get("api_key", "")

    def set_provider(self, provider_key: str, url: str, api_key: str) -> None:
        self.providers.setdefault(provider_key, {})["url"] = url
        self.providers.setdefault(provider_key, {})["api_key"] = api_key

    def save(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pro_provider": self.pro_provider,
            "pro_model": self.pro_model,
            "flash_provider": self.flash_provider,
            "flash_model": self.flash_model,
            "providers": self.providers,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    @classmethod
    def load(cls) -> "MimoConfig":
        path = _config_path()
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return cls()

        cfg = cls()
        cfg.pro_provider = data.get("pro_provider", cfg.pro_provider)
        cfg.pro_model = data.get("pro_model", cfg.pro_model)
        cfg.flash_provider = data.get("flash_provider", cfg.flash_provider)
        cfg.flash_model = data.get("flash_model", cfg.flash_model)
        cfg.providers = data.get("providers", {})
        return cfg
