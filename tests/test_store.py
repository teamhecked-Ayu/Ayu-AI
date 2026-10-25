import tempfile
import os
import sqlite3
import unittest
from pathlib import Path

from ayu.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "db" / "ayu.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_history_is_session_scoped_and_chronological(self):
        self.store.add_message("a", "user", "first")
        self.store.add_message("b", "user", "other")
        self.store.add_message("a", "assistant", "second")
        self.assertEqual(self.store.history("a"), [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ])

    def test_memories_can_be_added_searched_and_deleted(self):
        memory_id = self.store.remember("Use concise answers")
        self.store.remember("Use Python examples")
        self.assertEqual([r["content"] for r in self.store.recall("concise")], ["Use concise answers"])
        self.assertTrue(self.store.forget(memory_id))
        self.assertFalse(self.store.forget(memory_id))

    def test_search_treats_percent_as_literal(self):
        self.store.remember("100% complete")
        self.store.remember("100 things")
        self.assertEqual([r["content"] for r in self.store.recall("%")], ["100% complete"])

    def test_counts(self):
        self.store.add_message("a", "user", "hi")
        self.store.remember("pref")
        self.assertEqual(self.store.counts(), (1, 1))

    def test_additive_migration_preserves_legacy_memory_rows(self):
        legacy=Path(self.temp.name)/"legacy.sqlite3"
        with sqlite3.connect(legacy) as db:
            db.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL); CREATE TABLE memories(id INTEGER PRIMARY KEY,content TEXT NOT NULL,created_at TEXT NOT NULL); INSERT INTO memories(content,created_at) VALUES('old memory','2025-01-01T00:00:00+00:00');")
        migrated=Store(legacy)
        rows=migrated.recall("old memory")
        self.assertEqual(rows[0]["category"],"semantic")
        self.assertEqual(rows[0]["provenance"],"user")

    @unittest.skipIf(os.name=="nt","symlink behavior varies on Windows")
    def test_symlink_database_path_is_rejected(self):
        real=Path(self.temp.name)/"real.sqlite3"; alias=Path(self.temp.name)/"alias.sqlite3"
        Store(real); alias.symlink_to(real)
        with self.assertRaisesRegex(OSError,"symbolic link"):
            Store(alias)


if __name__ == "__main__":
    unittest.main()
