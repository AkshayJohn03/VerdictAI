"""Runtime configuration via pydantic-settings.

Every field maps to an environment variable with the ``VERDICTAI_`` prefix
(for example ``VERDICTAI_API_KEY``) and can also live in an optional ``.env``
file at the working directory. With no API key the library stays fully
offline: deterministic mock clients and heuristic judges are used instead.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VerdictAI runtime settings."""

    model_config = SettingsConfigDict(
        env_prefix="VERDICTAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5

    @property
    def has_api_key(self) -> bool:
        """True when an API key is configured (i.e. live-LLM mode is possible)."""
        return bool(self.api_key.strip())
