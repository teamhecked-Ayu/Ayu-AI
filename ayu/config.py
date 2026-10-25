"""Legacy TOML configuration helpers retained for backward compatibility."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

APP_DIR = Path.home() / ".ayu"
DEFAULT_CONFIG_PATH = APP_DIR / "config.toml"
DEFAULT_MEMORY_PATH = APP_DIR / "memory.db"
DEFAULT_PROJECTS_PATH = APP_DIR / "projects.json"


def default_config() -> dict[str, Any]:
    return {
        "mode": "basic",
        "offline": True,
        "workspace": str(Path.home() / "ayu-workspace"),
        "model": {"name": "tiny", "path": "", "backend": "local", "context_length": 256, "max_tokens": 64},
        "runtime": {"allow_destructive": False, "require_confirmation": True, "log_dir": str(APP_DIR / "logs"), "memory_path": str(DEFAULT_MEMORY_PATH), "projects_path": str(DEFAULT_PROJECTS_PATH)},
        "research": {"checkpoint_dir": str(APP_DIR / "checkpoints"), "data_dir": str(APP_DIR / "data"), "resume_training": False},
    }


def ensure_runtime_dirs() -> dict[str, Path]:
    dirs = {"app_dir": APP_DIR, "log_dir": APP_DIR / "logs", "checkpoint_dir": APP_DIR / "checkpoints", "data_dir": APP_DIR / "data", "workspace": Path(default_config()["workspace"])}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return dirs


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config = default_config()
    if target.exists():
        try:
            with target.open("rb") as handle:
                loaded = tomllib.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read TOML configuration at {target}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration at {target} must contain a TOML table")
        config = merge_dicts(config, loaded)
    return config


def _toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key, ensure_ascii=False)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, str): return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int): return str(value)
    if isinstance(value, float):
        if not math.isfinite(value): raise ValueError("TOML configuration cannot contain NaN or infinity")
        return repr(value)
    if isinstance(value, (list, tuple)): return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value).__name__}")


def _toml_dumps(obj: Any, indent: int = 0) -> str:
    if not isinstance(obj, dict): raise TypeError("Top-level TOML configuration must be a dictionary")
    lines: list[str] = []
    def emit(table: dict[str, Any], prefix: tuple[str, ...]) -> None:
        scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        children = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        if prefix:
            if lines: lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in prefix) + "]")
        for key, value in scalars:
            lines.append(f"{_toml_key(str(key))} = {_toml_scalar(value)}")
        for key, value in children:
            emit(value, prefix + (str(key),))
    emit(obj, ())
    return "\n".join(lines) + "\n"


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    if not isinstance(config, dict): raise TypeError("config must be a dictionary")
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix="ayu-config-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_toml_dumps(config))
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt": os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        try: os.unlink(temporary)
        except OSError: pass
        raise
    return target


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def init_project_state() -> dict[str, Any]:
    ensure_runtime_dirs()
    config = load_config()
    path = Path(config["runtime"]["projects_path"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text("[]", encoding="utf-8")
    return config

# Backward-compatible exports; implementations live in the status module.
def get_status():
    from .system import get_status as implementation
    return implementation()


def validate_config():
    from .system import validate_config as implementation
    return implementation()


def set_mode(mode: str):
    from .system import set_mode as implementation
    return implementation(mode)


def show_config():
    from .system import show_config as implementation
    return implementation()
