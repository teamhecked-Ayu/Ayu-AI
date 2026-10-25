"""Local-first runtime settings from environment variables and optional .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process environment."""
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and key.replace("_", "").isalnum() and key not in os.environ:
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ[key] = value
    except OSError:
        return


def _env_bool(name: str, default: str) -> bool:
    raw = os.getenv(name, default).strip().lower()
    if raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be true or false")
    return raw in {"true", "1", "yes"}


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model: str | None
    database_path: Path
    request_timeout: float = 90.0
    max_output_tokens: int = 2048
    mode: str = "local"
    local_model_path: Path | None = None
    llama_cli: str = "llama-cli"
    local_timeout: float = 300.0
    memory_enabled: bool = True
    remote_provider: str = "openai-compatible"
    newsapi_api_key: str | None = None
    newsapi_enabled: bool = False

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        _load_dotenv(env_file or Path.cwd() / ".env")
        base_url = os.getenv("AYU_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        timeout = float(os.getenv("AYU_REQUEST_TIMEOUT", "90"))
        max_tokens = int(os.getenv("AYU_MAX_OUTPUT_TOKENS", "512"))
        local_timeout = float(os.getenv("AYU_LOCAL_TIMEOUT", "300"))
        if not 1 <= timeout <= 600:
            raise ValueError("AYU_REQUEST_TIMEOUT must be between 1 and 600 seconds")
        if not 1 <= local_timeout <= 3600:
            raise ValueError("AYU_LOCAL_TIMEOUT must be between 1 and 3600 seconds")
        if not 1 <= max_tokens <= 8192:
            raise ValueError("AYU_MAX_OUTPUT_TOKENS must be between 1 and 8192")
        mode = os.getenv("AYU_MODE", "local").strip().lower()
        if mode not in {"basic", "local", "remote"}:
            raise ValueError("AYU_MODE must be basic, local, or remote; AyuLM research uses separate training/generation commands")
        remote_provider = os.getenv("AYU_PROVIDER", "openai-compatible").strip().lower()
        if remote_provider not in {"openai-compatible", "openai", "anthropic"}:
            raise ValueError("AYU_PROVIDER must be openai-compatible, openai, or anthropic")
        memory_enabled = _env_bool("AYU_MEMORY_ENABLED", "true")
        newsapi_enabled = _env_bool("AYU_NEWSAPI_ENABLED", "false")
        raw_model = os.getenv("AYU_MODEL", "").strip()
        if not raw_model:
            model_env = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL"}.get(remote_provider)
            raw_model = os.getenv(model_env, "").strip() if model_env else ""
        if remote_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif remote_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            api_key = os.getenv("AYU_API_KEY")
        raw_local_model = os.getenv("AYU_LOCAL_MODEL", "").strip()
        model_path = Path(raw_local_model).expanduser() if raw_local_model else None
        raw_database = os.getenv("AYU_DATABASE")
        db_path = Path(raw_database).expanduser() if raw_database else Path.home() / ".ayu-ai" / "ayu.sqlite3"
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=raw_model or None,
            database_path=db_path,
            request_timeout=timeout,
            max_output_tokens=max_tokens,
            mode=mode,
            local_model_path=model_path,
            llama_cli=os.getenv("AYU_LLAMA_CLI", "llama-cli"),
            local_timeout=local_timeout,
            memory_enabled=memory_enabled,
            remote_provider=remote_provider,
            newsapi_api_key=os.getenv("NEWSAPI_API_KEY"),
            newsapi_enabled=newsapi_enabled,
        )
