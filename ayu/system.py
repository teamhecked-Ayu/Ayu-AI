from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ayu.config import default_config, ensure_runtime_dirs, load_config, save_config


@dataclass
class SystemStatus:
    mode: str
    offline: bool
    workspace: str
    config_path: str
    memory_path: str
    model_name: str
    model_path: str
    backend: str


def get_status() -> SystemStatus:
    config = load_config()
    ensure_runtime_dirs()
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    return SystemStatus(
        mode=config["mode"],
        offline=bool(config["offline"]),
        workspace=config["workspace"],
        config_path=str(Path.home() / ".ayu" / "config.toml"),
        memory_path=str(runtime_cfg["memory_path"]),
        model_name=model_cfg["name"],
        model_path=str(model_cfg["path"]),
        backend=model_cfg["backend"],
    )


def validate_config() -> dict[str, Any]:
    cfg = load_config()
    errors: list[str] = []
    workspace = Path(cfg["workspace"])
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    if cfg["offline"] and cfg["model"]["path"]:
        if not Path(cfg["model"]["path"]).exists():
            errors.append(f"Configured model path does not exist: {cfg['model']['path']}")
    if cfg["mode"] not in {"basic", "local", "research"}:
        errors.append(f"Unsupported mode: {cfg['mode']}")
    return {"ok": not errors, "errors": errors, "config": cfg}


def set_mode(mode: str) -> dict[str, Any]:
    if mode not in {"basic", "local", "research"}:
        raise ValueError("Legacy configuration mode must be basic, local, or research")
    config = load_config()
    config["mode"] = mode
    save_config(config)
    return {"mode": mode, "config_path": str(Path.home() / ".ayu" / "config.toml")}


def show_config() -> dict[str, Any]:
    return load_config()
