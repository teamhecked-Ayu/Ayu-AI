from __future__ import annotations

import json
import sqlite3
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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
        target.parent.mkdir(parents=True, exist_ok=True)
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
        target.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return target


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
