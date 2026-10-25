from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from ayu.memory import MemoryStore
from ayu.project import ProjectManager


class LegacyCompatibilityTests(unittest.TestCase):
    def test_memory_round_trip_search_delete_and_private_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = MemoryStore(root / "memory.db")
            record = store.add("project", "demo", "Ayu project memory", source="test")
            self.assertIsNotNone(record.id)
            self.assertEqual(store.search("AYU")[0].id, record.id)
            self.assertEqual(store.status()["count"], 1)
            export = root / "export.json"
            store.export_json(export)
            self.assertEqual(json.loads(export.read_text(encoding="utf-8"))[0]["content"], "Ayu project memory")
            with self.assertRaisesRegex(ValueError, "active database"):
                store.export_json(root / "memory.db")
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE((root / "memory.db").stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(export.stat().st_mode), 0o600)
            self.assertTrue(store.delete(record.id))
            self.assertEqual(store.status()["count"], 0)

    def test_project_manager_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = ProjectManager(root / "projects.json")
            created = manager.create_project("demo", root / "workspace", ["ship tests"])
            self.assertEqual(created["goals"], ["ship tests"])
            self.assertEqual(manager.inspect_project("demo")["root"], str(root / "workspace"))
            self.assertEqual(manager.resume_project("demo")["status"], "resumed")


if __name__ == "__main__":
    unittest.main()
