from __future__ import annotations

import tempfile
from pathlib import Path

from ayu.config import default_config, load_config, save_config
from ayu.memory import MemoryStore


def test_default_config_shape() -> None:
    cfg = default_config()
    assert cfg["mode"] == "basic"
    assert cfg["offline"] is True
    assert "model" in cfg


def test_save_and_load_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.toml"
        save_config({"mode": "local", "offline": True}, path)
        loaded = load_config(path)
        assert loaded["mode"] == "local"


def test_memory_store_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(Path(tmpdir) / "memory.db")
        record = store.add("project", "demo", "AYU project memory", source="test")
        assert record.id is not None
        assert store.search("AYU")[0].id == record.id
        assert store.status()["count"] == 1
        assert store.delete(record.id) is True
