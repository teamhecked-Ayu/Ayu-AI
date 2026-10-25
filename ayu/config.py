from __future__ import annotations

import json
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
        "model": {
            "name": "tiny",
            "path": "",
            "backend": "local",
            "context_length": 256,
            "max_tokens": 64,
        },
        "runtime": {
            "allow_destructive": False,
            "require_confirmation": True,
            "log_dir": str(APP_DIR / "logs"),
            "memory_path": str(DEFAULT_MEMORY_PATH),
            "projects_path": str(DEFAULT_PROJECTS_PATH),
        },
        "research": {
            "checkpoint_dir": str(APP_DIR / "checkpoints"),
            "data_dir": str(APP_DIR / "data"),
            "resume_training": False,
        },
    }


def ensure_runtime_dirs() -> dict[str, Path]:
    dirs = {
        "app_dir": APP_DIR,
        "log_dir": APP_DIR / "logs",
        "checkpoint_dir": APP_DIR / "checkpoints",
        "data_dir": APP_DIR / "data",
        "workspace": Path(default_config()["workspace"]),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config = default_config()
    if target.exists():
        try:
            with target.open("rb") as handle:
                loaded = tomllib.load(handle)
            config = merge_dicts(config, loaded)
        except Exception:
            config = default_config()
    return config


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(_toml_dumps(config))
    return target


def _toml_dumps(obj: Any, indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{pad}[{key!r}]" if False else f"{pad}{key} = {{")
                lines.append(_toml_dumps(value, indent + 1))
                lines.append(f"{pad}}}")
            else:
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    rendered = str(value)
                else:
                    rendered = json.dumps(str(value))
                lines.append(f"{pad}{key} = {rendered}")
        return "\n".join(lines)
    return ""


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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
    return config
