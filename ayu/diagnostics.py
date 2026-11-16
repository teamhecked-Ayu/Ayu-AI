"""Privacy-safe, offline platform capability diagnostics."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from .settings import Settings


def _writable_directory(path: Path) -> bool:
    """Probe write access with a temporary file that is always removed."""
    if not path.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(prefix=".ayu-doctor-", dir=path, delete=True):
            pass
        return True
    except OSError:
        return False


def collect_platform_diagnostics(settings: Settings) -> dict[str, Any]:
    """Return local platform facts; never contacts providers or exposes secret values."""
    system = platform.system() or "unknown"
    machine = platform.machine() or "unknown"
    is_android = system == "Android" or hasattr(sys, "getandroidapilevel") or bool(os.environ.get("ANDROID_ROOT"))
    prefix = os.environ.get("PREFIX", "").lower()
    is_termux = is_android and (
        bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in prefix
    )
    profile = "android-termux" if is_termux else "android" if is_android else system.lower()

    database_parent = settings.database_path.parent
    writable = _writable_directory(database_parent)
    local_model_present = bool(settings.local_model_path and settings.local_model_path.is_file())
    if settings.local_backend == "llama-cli":
        runtime_available = bool(shutil.which(settings.llama_cli) or Path(settings.llama_cli).is_file())
        runtime_detail = "executable-detected" if runtime_available else "executable-not-found"
    else:
        runtime_available = True
        runtime_detail = "endpoint-configured-not-contacted"

    disk_free_mb = None
    try:
        disk_free_mb = round(shutil.disk_usage(database_parent if database_parent.is_dir() else Path.cwd()).free / (1024 * 1024))
    except OSError:
        pass

    checks = [
        {"id": "python-version", "status": "pass" if sys.version_info >= (3, 10) else "fail",
         "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
        {"id": "database-directory", "status": "pass" if writable else "fail",
         "detail": "writable" if writable else "missing-or-not-writable"},
        {"id": "local-runtime", "status": "pass" if runtime_available else "warning", "detail": runtime_detail},
        {"id": "local-model", "status": "pass" if local_model_present else "warning",
         "detail": "configured-file-present" if local_model_present else "not-configured-or-not-a-file"},
    ]
    if settings.local_backend in {"llama-server", "ollama", "vllm"}:
        checks.append({"id": "local-server", "status": "info", "detail": "configured-but-not-contacted"})
    if settings.mode == "remote":
        checks.append({"id": "remote-mode", "status": "info",
                       "detail": "selected-explicitly; credentials-and-endpoint-not-tested"})

    failed = any(item["status"] == "fail" for item in checks)
    warnings = any(item["status"] == "warning" for item in checks)
    return {
        "schema_version": "ayu-platform-diagnostics-v1",
        "status": "fail" if failed else "warning" if warnings else "ready",
        "platform": {
            "profile": profile,
            "system": system,
            "machine": machine,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "sqlite_version": sqlite3.sqlite_version,
        },
        "runtime": {
            "mode": settings.mode,
            "local_backend": settings.local_backend,
            "local_runtime": runtime_detail,
            "server_contacted": False,
            "local_model_file_present": local_model_present,
            "model_id_configured": bool(settings.local_model_id),
            "remote_provider_configured": bool(settings.api_key and settings.model),
            "remote_selected": settings.mode == "remote",
        },
        "capabilities": {
            "docker_executable_detected": bool(shutil.which("docker")),
            "pytorch_installed": importlib.util.find_spec("torch") is not None,
            "numpy_installed": importlib.util.find_spec("numpy") is not None,
            "disk_free_mb": disk_free_mb,
        },
        "checks": checks,
        "privacy": {
            "network_requests": 0,
            "secrets_included": False,
            "absolute_paths_included": False,
            "temporary_write_probe_removed": True,
        },
        "limitations": [
            "This report does not contact a model server or verify live inference.",
            "Executable detection does not prove Docker daemon health or sandbox security.",
            "Android and Termux support must be tested on the actual device and installed package set.",
            "Hardware acceleration, available RAM, model quality, and model provenance are not verified.",
        ],
    }
