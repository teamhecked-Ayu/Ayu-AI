# Copyright 2024 GenesisAI Contributors
# Licensed under the Apache License, Version 2.0


Centralized, immutable constant definitions for the GenesisAI runtime.

This module is the single source of truth for values that must remain
identical across every subsystem (app, core, engine, runtime, brain,
cognition, memory, llm, reasoning, cli, execution, ai_security,
benchmarking_suite, updater_server, crash_reporting, and all other
top-level packages). No subsystem may redefine or shadow any name
declared here; subsystems that need a derived or specialized value must
import it from this module and build upon it.

This module MUST NOT import from any other first-party GenesisAI
package. It may only depend on the Python standard library. This
constraint exists so that every other module in the codebase can safely
import `core.constants` during the earliest phase of interpreter
start-up, before logging, configuration, or dependency injection have
been initialized.


from __future__ import annotations

import enum
import os
import sys
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Application identity
# --------------------------------------------------------------------------

APP_NAME: Final[str] = "GenesisAI"
APP_SLUG: Final[str] = "genesisai"
APP_PACKAGE: Final[str] = "genesisai"
ORG_NAME: Final[str] = "GenesisAI Contributors"
APP_DESCRIPTION: Final[str] = (
    "GenesisAI: a terminal-first, production-grade Artificial "
    "Intelligence Operating System."
)
APP_HOMEPAGE: Final[str] = "https://github.com/genesisai-project/genesisai"
APP_LICENSE: Final[str] = "Apache-2.0"

# Fallback version used only when both the installed package metadata
# and the repository VERSION file are unavailable (e.g. an uninstalled
# checkout with no packaging metadata present). core.version should
# always be preferred over this constant at runtime.
FALLBACK_VERSION: Final[str] = "0.1.0"

# --------------------------------------------------------------------------
# Environment variable names
# --------------------------------------------------------------------------
# Every environment variable GenesisAI defines itself is namespaced with
# this prefix so the application never collides with unrelated host
# variables. Third-party provider SDK credential variables (below) are
# the only intentional exception, since those names are fixed by the
# providers themselves.

ENV_PREFIX: Final[str] = "GENESISAI_"

ENV_VAR_HOME: Final[str] = f"{ENV_PREFIX}HOME"
ENV_VAR_CONFIG_PATH: Final[str] = f"{ENV_PREFIX}CONFIG_PATH"
ENV_VAR_ENVIRONMENT: Final[str] = f"{ENV_PREFIX}ENVIRONMENT"
ENV_VAR_LOG_LEVEL: Final[str] = f"{ENV_PREFIX}LOG_LEVEL"
ENV_VAR_LOG_FORMAT: Final[str] = f"{ENV_PREFIX}LOG_FORMAT"
ENV_VAR_NO_COLOR: Final[str] = "NO_COLOR"
ENV_VAR_DEBUG: Final[str] = f"{ENV_PREFIX}DEBUG"
ENV_VAR_DATA_DIR: Final[str] = f"{ENV_PREFIX}DATA_DIR"
ENV_VAR_CACHE_DIR: Final[str] = f"{ENV_PREFIX}CACHE_DIR"
ENV_VAR_PLUGIN_PATH: Final[str] = f"{ENV_PREFIX}PLUGIN_PATH"
ENV_VAR_TELEMETRY_OPT_OUT: Final[str] = f"{ENV_PREFIX}TELEMETRY_OPT_OUT"
ENV_VAR_DEFAULT_PROVIDER: Final[str] = f"{ENV_PREFIX}DEFAULT_PROVIDER"
ENV_VAR_DEFAULT_MODEL: Final[str] = f"{ENV_PREFIX}DEFAULT_MODEL"

# Provider credential environment variables (read by llm/<provider>.py).
ENV_VAR_OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"
ENV_VAR_ANTHROPIC_API_KEY: Final[str] = "ANTHROPIC_API_KEY"
ENV_VAR_GOOGLE_API_KEY: Final[str] = "GOOGLE_API_KEY"
ENV_VAR_DEEPSEEK_API_KEY: Final[str] = "DEEPSEEK_API_KEY"
ENV_VAR_MISTRAL_API_KEY: Final[str] = "MISTRAL_API_KEY"
ENV_VAR_GROQ_API_KEY: Final[str] = "GROQ_API_KEY"
ENV_VAR_COHERE_API_KEY: Final[str] = "COHERE_API_KEY"
ENV_VAR_HUGGINGFACE_API_KEY: Final[str] = "HUGGINGFACE_API_KEY"
ENV_VAR_OPENROUTER_API_KEY: Final[str] = "OPENROUTER_API_KEY"
ENV_VAR_TOGETHER_API_KEY: Final[str] = "TOGETHER_API_KEY"
ENV_VAR_FIREWORKS_API_KEY: Final[str] = "FIREWORKS_API_KEY"
ENV_VAR_PERPLEXITY_API_KEY: Final[str] = "PERPLEXITY_API_KEY"
ENV_VAR_XAI_API_KEY: Final[str] = "XAI_API_KEY"
ENV_VAR_AZURE_OPENAI_API_KEY: Final[str] = "AZURE_OPENAI_API_KEY"
ENV_VAR_AZURE_OPENAI_ENDPOINT: Final[str] = "AZURE_OPENAI_ENDPOINT"
ENV_VAR_OLLAMA_HOST: Final[str] = "OLLAMA_HOST"

# --------------------------------------------------------------------------
# Runtime environment / platform enumerations
# --------------------------------------------------------------------------


class Environment(str, enum.Enum):
    """Named deployment/runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def default(cls) -> "Environment":
        return cls.PRODUCTION

    @classmethod
    def from_string(cls, value: str | None) -> "Environment":
        if not value:
            return cls.default()
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.default()


class Platform(str, enum.Enum):
    """Operating system platforms officially supported by GenesisAI."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    ANDROID_TERMUX = "android_termux"
    UNKNOWN = "unknown"

    @classmethod
    def detect(cls) -> "Platform":
        if "ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ:
            return cls.ANDROID_TERMUX
        platform_name = sys.platform
        if platform_name.startswith("linux"):
            return cls.LINUX
        if platform_name == "darwin":
            return cls.MACOS
        if platform_name.startswith("win"):
            return cls.WINDOWS
        return cls.UNKNOWN


class LogLevel(str, enum.Enum):
    """Canonical logging severities used by core.logger across GenesisAI."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def default(cls) -> "LogLevel":
        return cls.INFO


class LogFormat(str, enum.Enum):
    """Output encodings supported by the structured logger."""

    TEXT = "text"
    JSON = "json"
    RICH = "rich"


class ExitCode(enum.IntEnum):
    """Process exit codes returned by the `genesisai` CLI entry point."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    CONFIGURATION_ERROR = 2
    DEPENDENCY_ERROR = 3
    VALIDATION_ERROR = 4
    PERMISSION_ERROR = 5
    NETWORK_ERROR = 6
    PROVIDER_ERROR = 7
    TIMEOUT_ERROR = 8
    NOT_FOUND_ERROR = 9
    PLUGIN_ERROR = 10
    SECURITY_ERROR = 11
    INTERRUPTED = 130  # Conventional SIGINT exit code (128 + 2)


# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------

_PLATFORM: Final[Platform] = Platform.detect()


def _resolve_home_directory() -> Path:
    """Resolve the GenesisAI home directory, honoring the override env var."""
    override = os.environ.get(ENV_VAR_HOME)
    if override:
        return Path(override).expanduser().resolve()

    if _PLATFORM == Platform.WINDOWS:
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if _PLATFORM == Platform.MACOS:
        return Path.home() / "Library" / "Application Support" / APP_NAME
    # Linux, Android/Termux, and unknown POSIX-like platforms follow the
    # XDG Base Directory convention with a dotfile fallback.
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / APP_SLUG
    return Path.home() / f".{APP_SLUG}"


HOME_DIR: Final[Path] = _resolve_home_directory()
CONFIG_DIR: Final[Path] = HOME_DIR / "config"
DATA_DIR: Final[Path] = Path(os.environ.get(ENV_VAR_DATA_DIR, str(HOME_DIR / "data")))
CACHE_DIR: Final[Path] = Path(
    os.environ.get(ENV_VAR_CACHE_DIR, str(HOME_DIR / "cache"))
)
LOG_DIR: Final[Path] = HOME_DIR / "logs"
PLUGIN_DIR: Final[Path] = HOME_DIR / "plugins"
MEMORY_DIR: Final[Path] = DATA_DIR / "memory"
SESSIONS_DIR: Final[Path] = DATA_DIR / "sessions"
CHECKPOINTS_DIR: Final[Path] = DATA_DIR / "checkpoints"
WORKSPACES_DIR: Final[Path] = HOME_DIR / "workspaces"

DEFAULT_CONFIG_FILENAMES: Final[tuple[str, ...]] = (
    "genesisai.toml",
    "genesisai.yaml",
    "genesisai.yml",
    "genesisai.json",
)

ALL_MANAGED_DIRECTORIES: Final[tuple[Path, ...]] = (
    HOME_DIR,
    CONFIG_DIR,
    DATA_DIR,
    CACHE_DIR,
    LOG_DIR,
    PLUGIN_DIR,
    MEMORY_DIR,
    SESSIONS_DIR,
    CHECKPOINTS_DIR,
    WORKSPACES_DIR,
)

# --------------------------------------------------------------------------
# Encoding / formatting defaults
# --------------------------------------------------------------------------

DEFAULT_ENCODING: Final[str] = "utf-8"
DEFAULT_NEWLINE: Final[str] = "\n"
DEFAULT_LOCALE: Final[str] = "en_US.UTF-8"
DEFAULT_TIMEZONE: Final[str] = "UTC"
DEFAULT_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%fZ"

# --------------------------------------------------------------------------
# Networking / I/O defaults
# --------------------------------------------------------------------------

DEFAULT_CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_READ_TIMEOUT_SECONDS: Final[float] = 120.0
DEFAULT_TOTAL_TIMEOUT_SECONDS: Final[float] = 300.0
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_BACKOFF_BASE_SECONDS: Final[float] = 0.5
DEFAULT_RETRY_BACKOFF_MAX_SECONDS: Final[float] = 30.0
DEFAULT_MAX_CONCURRENT_REQUESTS: Final[int] = 8
DEFAULT_CHUNK_SIZE_BYTES: Final[int] = 65536
DEFAULT_MAX_RESPONSE_SIZE_BYTES: Final[int] = 64 * 1024 * 1024  # 64 MiB

# --------------------------------------------------------------------------
# LLM subsystem defaults (overridable by core.configuration)
# --------------------------------------------------------------------------

DEFAULT_PROVIDER_NAME: Final[str] = "anthropic"
DEFAULT_MODEL_NAME: Final[str] = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 1.0
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 4096
DEFAULT_CONTEXT_WINDOW_TOKENS: Final[int] = 128_000
SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = (
    "openai",
    "anthropic",
    "gemini",
    "deepseek",
    "mistral",
    "groq",
    "cohere",
    "ollama",
    "huggingface",
    "openrouter",
    "together",
    "fireworks",
    "perplexity",
    "xai",
    "azure_openai",
    "vertex_ai",
    "bedrock",
    "local",
)

# --------------------------------------------------------------------------
# Concurrency / scheduling defaults
# --------------------------------------------------------------------------

DEFAULT_WORKER_POOL_SIZE: Final[int] = max(2, (os.cpu_count() or 4))
DEFAULT_TASK_QUEUE_MAXSIZE: Final[int] = 1000
DEFAULT_SHUTDOWN_GRACE_PERIOD_SECONDS: Final[float] = 15.0
DEFAULT_HEALTHCHECK_INTERVAL_SECONDS: Final[float] = 30.0
DEFAULT_WATCHDOG_TIMEOUT_SECONDS: Final[float] = 60.0

# --------------------------------------------------------------------------
# Memory subsystem defaults
# --------------------------------------------------------------------------

DEFAULT_SHORT_TERM_MEMORY_CAPACITY: Final[int] = 50
DEFAULT_WORKING_MEMORY_CAPACITY: Final[int] = 20
DEFAULT_LONG_TERM_MEMORY_RETENTION_DAYS: Final[int] = 365
DEFAULT_EMBEDDING_DIMENSIONS: Final[int] = 1536
DEFAULT_VECTOR_SIMILARITY_TOP_K: Final[int] = 8

# --------------------------------------------------------------------------
# Security defaults
# --------------------------------------------------------------------------

DEFAULT_SECRET_REDACTION_TOKEN: Final[str] = "********"
DEFAULT_SESSION_TOKEN_BYTES: Final[int] = 32
DEFAULT_PASSWORD_HASH_ALGORITHM: Final[str] = "argon2id"
DEFAULT_FILE_PERMISSIONS_SECRET: Final[int] = 0o600
DEFAULT_DIR_PERMISSIONS_SECRET: Final[int] = 0o700

# --------------------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------------------

CURRENT_PLATFORM: Final[Platform] = _PLATFORM
IS_INTERACTIVE_TTY: Final[bool] = sys.stdin.isatty() and sys.stdout.isatty()
MIN_PYTHON_VERSION: Final[tuple[int, int]] = (3, 10)

__all__ = [
    "APP_NAME",
    "APP_SLUG",
    "APP_PACKAGE",
    "ORG_NAME",
    "APP_DESCRIPTION",
    "APP_HOMEPAGE",
    "APP_LICENSE",
    "FALLBACK_VERSION",
    "ENV_PREFIX",
    "ENV_VAR_HOME",
    "ENV_VAR_CONFIG_PATH",
    "ENV_VAR_ENVIRONMENT",
    "ENV_VAR_LOG_LEVEL",
    "ENV_VAR_LOG_FORMAT",
    "ENV_VAR_NO_COLOR",
    "ENV_VAR_DEBUG",
    "ENV_VAR_DATA_DIR",
    "ENV_VAR_CACHE_DIR",
    "ENV_VAR_PLUGIN_PATH",
    "ENV_VAR_TELEMETRY_OPT_OUT",
    "ENV_VAR_DEFAULT_PROVIDER",
    "ENV_VAR_DEFAULT_MODEL",
    "ENV_VAR_OPENAI_API_KEY",
    "ENV_VAR_ANTHROPIC_API_KEY",
    "ENV_VAR_GOOGLE_API_KEY",
    "ENV_VAR_DEEPSEEK_API_KEY",
    "ENV_VAR_MISTRAL_API_KEY",
    "ENV_VAR_GROQ_API_KEY",
    "ENV_VAR_COHERE_API_KEY",
    "ENV_VAR_HUGGINGFACE_API_KEY",
    "ENV_VAR_OPENROUTER_API_KEY",
    "ENV_VAR_TOGETHER_API_KEY",
    "ENV_VAR_FIREWORKS_API_KEY",
    "ENV_VAR_PERPLEXITY_API_KEY",
    "ENV_VAR_XAI_API_KEY",
    "ENV_VAR_AZURE_OPENAI_API_KEY",
    "ENV_VAR_AZURE_OPENAI_ENDPOINT",
    "ENV_VAR_OLLAMA_HOST",
    "Environment",
    "Platform",
    "LogLevel",
    "LogFormat",
    "ExitCode",
    "HOME_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "CACHE_DIR",
    "LOG_DIR",
    "PLUGIN_DIR",
    "MEMORY_DIR",
    "SESSIONS_DIR",
    "CHECKPOINTS_DIR",
    "WORKSPACES_DIR",
    "DEFAULT_CONFIG_FILENAMES",
    "ALL_MANAGED_DIRECTORIES",
    "DEFAULT_ENCODING",
    "DEFAULT_NEWLINE",
    "DEFAULT_LOCALE",
    "DEFAULT_TIMEZONE",
    "DEFAULT_DATETIME_FORMAT",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "DEFAULT_TOTAL_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF_BASE_SECONDS",
    "DEFAULT_RETRY_BACKOFF_MAX_SECONDS",
    "DEFAULT_MAX_CONCURRENT_REQUESTS",
    "DEFAULT_CHUNK_SIZE_BYTES",
    "DEFAULT_MAX_RESPONSE_SIZE_BYTES",
    "DEFAULT_PROVIDER_NAME",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_CONTEXT_WINDOW_TOKENS",
    "SUPPORTED_PROVIDERS",
    "DEFAULT_WORKER_POOL_SIZE",
    "DEFAULT_TASK_QUEUE_MAXSIZE",
    "DEFAULT_SHUTDOWN_GRACE_PERIOD_SECONDS",
    "DEFAULT_HEALTHCHECK_INTERVAL_SECONDS",
    "DEFAULT_WATCHDOG_TIMEOUT_SECONDS",
    "DEFAULT_SHORT_TERM_MEMORY_CAPACITY",
    "DEFAULT_WORKING_MEMORY_CAPACITY",
    "DEFAULT_LONG_TERM_MEMORY_RETENTION_DAYS",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_VECTOR_SIMILARITY_TOP_K",
    "DEFAULT_SECRET_REDACTION_TOKEN",
    "DEFAULT_SESSION_TOKEN_BYTES",
    "DEFAULT_PASSWORD_HASH_ALGORITHM",
    "DEFAULT_FILE_PERMISSIONS_SECRET",
    "DEFAULT_DIR_PERMISSIONS_SECRET",
    "CURRENT_PLATFORM",
    "IS_INTERACTIVE_TTY",
    "MIN_PYTHON_VERSION",
]
