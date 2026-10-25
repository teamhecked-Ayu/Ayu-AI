from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ayu.config import default_config


@dataclass
class MemoryRecord:
    id: int | None
    category: str
    key: str
    content: str
    source: str
    created_at: str
    updated_at: str
    metadata: str = "{}"


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path(default_config()["runtime"]["memory_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.db_path.is_symlink():
            raise OSError(f"Memory database path must not be a symbolic link: {self.db_path}")
        self._init_db()
        if os.name != "nt":
            self.db_path.chmod(0o600)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)"
            )
            connection.commit()

    def add(self, category: str, key: str, content: str, source: str = "user", metadata: dict[str, Any] | None = None) -> MemoryRecord:
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        created_at = _timestamp()
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (category, key, content, source, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (category, key, content, source, created_at, created_at, metadata_json),
            )
            connection.commit()
            memory_id = cursor.lastrowid
        return self.get(memory_id)

    def get(self, memory_id: int) -> MemoryRecord:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT id, category, key, content, source, created_at, updated_at, metadata FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Memory entry {memory_id} not found")
        return MemoryRecord(*row)

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        q = f"%{query.lower()}%"
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, category, key, content, source, created_at, updated_at, metadata
                FROM memories
                WHERE lower(category) LIKE ? OR lower(key) LIKE ? OR lower(content) LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (q, q, q, limit),
            ).fetchall()
        return [MemoryRecord(*row) for row in rows]

    def status(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
            categories = connection.execute(
                "SELECT category, COUNT(*) FROM memories GROUP BY category ORDER BY category"
            ).fetchall()
        return {"count": rows[0], "categories": {category: count for category, count in categories}}

    def delete(self, memory_id: int) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            connection.commit()
        return cursor.rowcount > 0

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        if target.resolve() == self.db_path.resolve():
            raise ValueError("Memory export destination cannot be the active database")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        rows = self.search("", limit=100000)
        items = [
            {
                "id": item.id,
                "category": item.category,
                "key": item.key,
                "content": item.content,
                "source": item.source,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "metadata": json.loads(item.metadata or "{}"),
            }
            for item in rows
        ]
        fd, temporary = tempfile.mkstemp(prefix="ayu-memory-export-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(items, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt": os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except BaseException:
            try: os.unlink(temporary)
            except OSError: pass
            raise
        return target


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
