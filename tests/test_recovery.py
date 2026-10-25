import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.cli import main
from ayu.store import Store


class RecoveryTests(unittest.TestCase):
    def test_backup_requires_new_destination_and_restore_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); db_path=root/"ayu.sqlite3"; backup=root/"backup.sqlite3"
            with patch.dict(os.environ,{"AYU_DATABASE":str(db_path)},clear=True):
                store=Store(db_path); store.remember("before backup",category="preference")
                self.assertEqual(main(["backup",str(backup)]),0)
                store.remember("after backup",category="semantic")
                err=io.StringIO()
                with contextlib.redirect_stderr(err): self.assertEqual(main(["restore",str(backup)]),2)
                self.assertIn("--confirm",err.getvalue())
                self.assertEqual(main(["restore",str(backup),"--confirm"]),0)
                rollback=list(root.glob("ayu.sqlite3.pre-restore-*.sqlite3"))
                self.assertEqual(len(rollback),1)
            restored=Store(db_path).all_memories()
            self.assertEqual([r["content"] for r in restored],["before backup"])

    @unittest.skipIf(os.name=="nt","POSIX permission bits do not map consistently on Windows")
    def test_database_is_created_with_private_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/"private.sqlite3"
            Store(db)
            self.assertEqual(stat.S_IMODE(db.stat().st_mode),0o600)

    def test_memory_export_import_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); db_path=root/"ayu.sqlite3"; export=root/"memories.json"
            with patch.dict(os.environ,{"AYU_DATABASE":str(db_path)},clear=True):
                store=Store(db_path); store.remember("keep local",category="preference")
                self.assertEqual(main(["memory","export",str(export)]),0)
                payload=json.loads(export.read_text(encoding="utf-8"))
                self.assertEqual(payload["format"],"ayu-memory-export-v1")
                self.assertEqual(main(["memory","import",str(export)]),0)
                self.assertEqual(len(Store(db_path).all_memories()),1)
                invalid=root/"invalid.json"
                invalid.write_text(json.dumps({"format":"ayu-memory-export-v1","memories":[{"content":"good","category":"semantic"},{"content":"bad","category":"made-up"}]}),encoding="utf-8")
                err=io.StringIO()
                with contextlib.redirect_stderr(err): self.assertEqual(main(["memory","import",str(invalid)]),1)
                self.assertEqual(len(Store(db_path).all_memories()),1)


if __name__=="__main__": unittest.main()
