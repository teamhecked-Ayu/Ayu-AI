from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.coding import CodeRunError, CodingRunner, apply_edits, sanitize_test_output
from ayu.cli import main
from ayu.sandbox import SandboxError
from ayu.store import Store


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.schemas = []

    def complete(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("unexpected provider call")
        return self.responses.pop(0)

    def complete_structured(self, messages, schema, *, schema_name):
        self.schemas.append((schema_name, schema))
        return self.complete(messages)


class FakeSandbox:
    """Test-only adapter; it exercises workflow behavior but provides no isolation."""
    def __init__(self, exit_codes=None):
        self.exit_codes = list(exit_codes or [0])

    def preflight(self):
        return {"backend": "fake-test-only", "policy_version": "unit-test", "network": "not-tested"}

    def run_tests(self, workspace, suite, paths, timeout, *, run_id, on_start=None):
        if on_start is not None:
            on_start({"backend": "fake-test-only", "container_name": "ayu-test-unit-00000000",
                      "image": "not-used", "policy_version": "unit-test"})
        exit_code = self.exit_codes.pop(0) if self.exit_codes else 0
        return {"command": ["synthetic-test-result"], "exit_code": exit_code, "timed_out": False,
                "duration_seconds": 0.0, "stdout": "fake test result: passed" if exit_code == 0 else "fake test result: failed",
                "stderr": "", "infrastructure_error": False, "output_truncated": False,
                "backend": "fake-test-only", "image": "not-used", "policy_version": "unit-test",
                "container_name": "ayu-test-unit-00000000", "cleanup_confirmed": True}


def init_repo(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Ayu Test"], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "ayu-test@example.invalid"], check=True, capture_output=True, env=env)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "--", *files.keys()], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, capture_output=True, env=env)


class PatchSafetyTests(unittest.TestCase):
    def test_exact_edit_succeeds_and_preserves_only_requested_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "a.py"
            target.write_text("value = 1\n", encoding="utf-8")
            before = target.read_bytes()
            changes = apply_edits(root, [{"path": "a.py", "old_text": "value = 1", "new_text": "value = 2"}], {"a.py": hashlib.sha256(before).hexdigest()})
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
            self.assertEqual(changes[0]["path"], "a.py")

    def test_traversal_ambiguous_context_and_stale_hash_rejected_without_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "a.py"
            target.write_text("x = 1\nx = 1\n", encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaises(CodeRunError):
                apply_edits(root, [{"path": "../a.py", "old_text": "x = 1", "new_text": "x = 2"}], {})
            with self.assertRaisesRegex(CodeRunError, "exactly once"):
                apply_edits(root, [{"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}], {"a.py": digest})
            with self.assertRaisesRegex(CodeRunError, "stale"):
                apply_edits(root, [{"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}], {"a.py": "0" * 64})
            self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\nx = 1\n")

    def test_multiple_nonoverlapping_patches_to_one_file_apply_as_one_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "a.py"
            target.write_text("x = 1\ny = 2\n", encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            changes = apply_edits(root, [
                {"path": "a.py", "old_text": "x = 1", "new_text": "x = 3"},
                {"path": "a.py", "old_text": "y = 2", "new_text": "y = 4"},
            ], {"a.py": digest})
            self.assertEqual(target.read_text(encoding="utf-8"), "x = 3\ny = 4\n")
            self.assertEqual(len(changes), 1)

    def test_crlf_file_accepts_lf_model_context_and_preserves_crlf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "a.py"
            original = b"x = 1\r\ny = 2\r\n"
            target.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            changes = apply_edits(root, [
                {"path": "a.py", "old_text": "x = 1\ny = 2", "new_text": "x = 3\ny = 4"}
            ], {"a.py": digest})
            self.assertEqual(target.read_bytes(), b"x = 3\r\ny = 4\r\n")
            self.assertEqual(changes[0]["path"], "a.py")

    def test_overlapping_same_file_contexts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "a.py"
            target.write_text("x = 1\ny = 2\n", encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaisesRegex(CodeRunError, "overlap"):
                apply_edits(root, [
                    {"path": "a.py", "old_text": "x = 1\ny = 2", "new_text": "z = 3"},
                    {"path": "a.py", "old_text": "y = 2", "new_text": "y = 4"},
                ], {"a.py": digest})
            self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\ny = 2\n")

    def test_overlapping_crlf_contexts_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "a.py"
            original = b"x = 1\r\ny = 2\r\n"
            target.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            with self.assertRaisesRegex(CodeRunError, "overlap"):
                apply_edits(root, [
                    {"path": "a.py", "old_text": "x = 1\ny = 2", "new_text": "z = 3"},
                    {"path": "a.py", "old_text": "y = 2", "new_text": "y = 4"},
                ], {"a.py": digest})
            self.assertEqual(target.read_bytes(), original)

    def test_audit_output_redacts_common_credentials_and_is_bounded(self):
        output = "x" * 2500 + "\nOPENAI_API_KEY=demo-placeholder-value Authorization: Bearer placeholder123"
        clean = sanitize_test_output(output, 2000)
        self.assertNotIn("demo-placeholder-value", clean)
        self.assertNotIn("placeholder123", clean)
        self.assertLessEqual(len(clean), 2000)


class CodingWorkflowTests(unittest.TestCase):
    def test_coding_module_parses_with_python_310_grammar(self):
        module = Path(__file__).resolve().parents[1] / "ayu" / "coding.py"
        source = module.read_text(encoding="utf-8")
        for minor in (10, 11):
            with self.subTest(minor=minor):
                ast.parse(source, filename=str(module), feature_version=(3, minor))

    def test_python_api_rejects_provider_without_structured_output(self):
        class PlainProvider:
            def complete(self, messages):
                return "{}"
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(CodeRunError, "strict JSON Schema output"):
                CodingRunner(PlainProvider(), Store(Path(temp) / "state.sqlite3"))

    def test_failed_patch_reports_persisted_run_id_for_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {"calc.py": "def add(a, b):\n    return a - b\n"})
            store = Store(Path(temp) / "state.sqlite3")
            provider = FakeProvider([
                json.dumps({"files_to_read": ["calc.py"], "success_criteria": ["add returns a sum"], "risk_notes": []}),
                json.dumps({"summary": "Bad context must fail safely", "edits": [{"path": "calc.py", "old_text": "not present", "new_text": "return a + b"}]}),
            ])
            with self.assertRaisesRegex(CodeRunError, "Exact edit context") as error:
                CodingRunner(provider, store, sandbox_runner=FakeSandbox()).run("Make add return the sum", root, suite="compileall")
            message = str(error.exception)
            run_id = message.split("Coding run ", 1)[1].split(" failed:", 1)[0]
            self.assertEqual(len(run_id), 32)
            self.assertEqual(store.get_agent_run(run_id)["status"], "failed")
            events = store.list_agent_events(run_id)
            failure = next(json.loads(item["payload_json"]) for item in events if item["event_type"] == "failure")
            worktree = Path(failure["worktree"])
            preparation = next(json.loads(item["payload_json"]) for item in events if item["event_type"] == "worktree_preparing")
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "branch", "-D", preparation["branch"]], check=True, capture_output=True)
            shutil.rmtree(worktree.parent, ignore_errors=True)

    def test_private_run_directory_failure_is_recorded_after_atomic_run_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {"calc.py": "value = 1\n"})
            store = Store(Path(temp) / "state.sqlite3")
            runner = CodingRunner(FakeProvider([]), store, sandbox_runner=FakeSandbox())
            real_mkdir = Path.mkdir

            def fail_run_directory(path, *args, **kwargs):
                if path.name.startswith("ayu-code-"):
                    raise OSError("synthetic directory creation failure")
                return real_mkdir(path, *args, **kwargs)

            with patch.object(Path, "mkdir", new=fail_run_directory):
                with self.assertRaisesRegex(CodeRunError, "synthetic directory creation failure"):
                    runner.run("make a safe code change", root, suite="compileall")
            with store.connect() as db:
                rows = db.execute("SELECT id,status FROM agent_runs").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            events = store.list_agent_events(rows[0]["id"])
            self.assertEqual(events[0]["event_type"], "run_started")
            self.assertEqual(events[-1]["event_type"], "failure")

    def test_terminated_child_run_is_recovered_as_unknown_after_database_reopen(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state.sqlite3"
            run_id = "e" * 32
            child_code = (
                "import sys,time; from pathlib import Path; from ayu.store import Store; "
                "Store(Path(sys.argv[1])).create_agent_run(sys.argv[2], 'interrupted fixture', 'local', 2, "
                "initial_event=('coding','run_started',{'run_id':sys.argv[2]})); time.sleep(60)"
            )
            process = subprocess.Popen([sys.executable, "-c", child_code, str(database), run_id],
                                       cwd=Path(__file__).resolve().parents[1],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                persisted = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        stderr = process.stderr.read().decode("utf-8", errors="replace")
                        self.fail(f"fixture child exited before persisting its run: {stderr[:1000]}")
                    if database.exists():
                        try:
                            persisted = Store(database).get_agent_run(run_id)
                        except sqlite3.OperationalError:
                            persisted = None
                        if persisted is not None:
                            break
                    time.sleep(0.05)
                self.assertIsNotNone(persisted, "child did not persist its run before deadline")
            finally:
                process.kill()
                process.wait(timeout=10)
                if process.stderr:
                    process.stderr.close()
            reopened = Store(database)
            self.assertEqual(reopened.get_agent_run(run_id)["status"], "running")
            self.assertEqual(reopened.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=True)["status"],
                             "unknown_outcome")
            self.assertEqual(len(reopened.list_agent_events(run_id)), 2)

    def test_json_parser_accepts_only_a_single_strict_json_fence(self):
        runner = CodingRunner(FakeProvider([]), Store(Path(tempfile.mkdtemp()) / "state.sqlite3"))
        parsed = runner._json('```json\n{"status":"supported","issues":[]}\n```', "Reviewer", {"status", "issues"})
        self.assertEqual(parsed["status"], "supported")
        with self.assertRaisesRegex(CodeRunError, "valid JSON"):
            runner._json('Here is the result: {"status":"supported","issues":[]}', "Reviewer", {"status", "issues"})

    def test_cli_requires_sandbox_before_provider_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"; root.mkdir()
            stderr = io.StringIO()
            env = {"AYU_MODE": "local", "AYU_LOCAL_BACKEND": "llama-server", "AYU_DATABASE": str(Path(temp) / "state.sqlite3")}
            class UnavailableSandbox:
                def preflight(self):
                    raise SandboxError("Docker CLI is unavailable")
            with patch.dict(os.environ, env, clear=True), patch("ayu.sandbox.DockerSandboxRunner.from_environment", return_value=UnavailableSandbox()), patch("ayu.cli.build_local_provider") as build, contextlib.redirect_stderr(stderr):
                result = main(["code", "run", "make a change", "--workspace", str(root), "--test-suite", "compileall"])
            self.assertEqual(result, 2)
            self.assertIn("Docker sandbox is unavailable", stderr.getvalue())
            build.assert_not_called()

    def test_cli_requires_llama_server_for_structured_coding(self):
        with tempfile.TemporaryDirectory() as temp:
            stderr = io.StringIO()
            env = {"AYU_MODE": "local", "AYU_LOCAL_BACKEND": "llama-cli", "AYU_DATABASE": str(Path(temp) / "state.sqlite3")}
            argv = ["code", "run", "make a change", "--workspace", temp, "--test-suite", "compileall"]
            with patch.dict(os.environ, env, clear=True), patch("ayu.cli.build_local_provider") as build, contextlib.redirect_stderr(stderr):
                result = main(argv)
            self.assertEqual(result, 2)
            self.assertIn("llama-server", stderr.getvalue())
            build.assert_not_called()

    def test_run_uses_clean_worktree_runs_tests_and_persists_diff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            original = "def add(a, b):\n    return a - b\n"
            init_repo(root, {
                "calc.py": original,
                "tests/test_calc.py": "import unittest\nfrom calc import add\n\nclass CalcTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n",
            })
            store = Store(Path(temp) / "state.sqlite3")
            provider = FakeProvider([
                json.dumps({"files_to_read": ["calc.py"], "success_criteria": ["add returns the sum"], "risk_notes": []}),
                json.dumps({"summary": "Correct addition implementation", "edits": [{"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}]}),
                json.dumps({"status": "supported", "issues": []}),
            ])
            result = CodingRunner(provider, store, sandbox_runner=FakeSandbox()).run(
                "Make add return the sum", root, suite="unittest", test_paths=["tests"], timeout=20
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["test"]["exit_code"], 0)
            self.assertEqual(len(provider.calls), 3)
            self.assertEqual((root / "calc.py").read_text(encoding="utf-8"), original)
            worktree = Path(result["worktree"])
            self.assertEqual((worktree / "calc.py").read_text(encoding="utf-8"), "def add(a, b):\n    return a + b\n")
            saved = store.get_agent_run(result["run_id"])
            self.assertEqual(saved["status"], "completed")
            events = store.list_agent_events(result["run_id"])
            self.assertIn("final_diff", [event["event_type"] for event in events])
            persisted = next(json.loads(event["payload_json"]) for event in events if event["event_type"] == "final_diff")
            self.assertIn("return a + b", persisted["diff"])
            test_evidence = next(json.loads(event["payload_json"]) for event in events if event["event_type"] == "test_result")
            self.assertEqual(len(test_evidence["stderr_capture_sha256"]), 64)
            self.assertIn("fake test result: passed", test_evidence["stdout_excerpt"])
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "branch", "-D", result["branch"]], check=True, capture_output=True)

    def test_single_correction_is_bounded_and_retested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {
                "calc.py": "def add(a, b):\n    return a - b\n",
                "tests/test_calc.py": "import unittest\nfrom calc import add\nclass CalcTests(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(2, 3), 5)\n",
            })
            store = Store(Path(temp) / "state.sqlite3")
            provider = FakeProvider([
                json.dumps({"files_to_read": ["calc.py"], "success_criteria": ["add returns sum"], "risk_notes": []}),
                json.dumps({"summary": "First attempt", "edits": [{"path": "calc.py", "old_text": "return a - b", "new_text": "return a * b"}]}),
                json.dumps({"summary": "Correct failing arithmetic", "edits": [{"path": "calc.py", "old_text": "return a * b", "new_text": "return a + b"}]}),
                json.dumps({"status": "supported", "issues": []}),
            ])
            result = CodingRunner(provider, store, sandbox_runner=FakeSandbox([1, 0])).run(
                "Make add return the sum", root, suite="unittest", timeout=20
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["test"]["exit_code"], 0)
            self.assertEqual(len(provider.calls), 4)
            types = [event["event_type"] for event in store.list_agent_events(result["run_id"])]
            self.assertEqual(types.count("correction"), 1)
            worktree = Path(result["worktree"])
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "branch", "-D", result["branch"]], check=True, capture_output=True)

    def test_dirty_source_repository_is_refused_before_run_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {"calc.py": "x = 1\n"})
            (root / "calc.py").write_text("x = 2\n", encoding="utf-8")
            store = Store(Path(temp) / "state.sqlite3")
            runner = CodingRunner(FakeProvider([]), store, sandbox_runner=FakeSandbox())
            with self.assertRaisesRegex(CodeRunError, "dirty repository"):
                runner.run("change it", root, suite="compileall")
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0], 0)

    def test_git_filter_attributes_are_rejected_before_worktree_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {".gitattributes": "*.py filter=untrusted\n", "calc.py": "x = 1\n"})
            store = Store(Path(temp) / "state.sqlite3")
            provider = FakeProvider([])
            runner = CodingRunner(provider, store, sandbox_runner=FakeSandbox())
            with self.assertRaisesRegex(CodeRunError, "content filter"):
                runner.run("make a code change", root, suite="compileall")
            self.assertEqual(provider.calls, [])
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0], 0)

    def test_running_coding_run_can_be_marked_unknown_without_rerun_or_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "state.sqlite3")
            run_id = "a" * 32
            store.create_agent_run(run_id, "test task", "local", 2,
                                   initial_event=("coding", "run_started", {"worktree": "synthetic"}))
            result = store.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=True)
            self.assertEqual(result, {"run_id": run_id, "status": "unknown_outcome", "changed": True})
            self.assertEqual(store.get_agent_run(run_id)["status"], "unknown_outcome")
            events = store.list_agent_events(run_id)
            self.assertEqual([event["event_type"] for event in events], ["run_started", "recovery_unknown_outcome"])
            recovery = json.loads(events[-1]["payload_json"])
            self.assertFalse(recovery["automatic_rerun"])
            self.assertTrue(recovery["worktree_preserved"])
            self.assertFalse(recovery["container_cleanup_attempted"])
            self.assertEqual(store.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=True)["changed"], False)
            self.assertEqual(len(store.list_agent_events(run_id)), 2)

    def test_unknown_outcome_recovery_requires_confirmation_running_state_and_coding_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "state.sqlite3")
            run_id = "b" * 32
            store.create_agent_run(run_id, "test task", "local", 2)
            with self.assertRaisesRegex(ValueError, "must confirm"):
                store.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=False)
            with self.assertRaisesRegex(ValueError, "no persisted coding events"):
                store.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=True)
            store.record_agent_event(run_id, 0, "planner", "plan", {})
            with self.assertRaisesRegex(ValueError, "no persisted coding events"):
                store.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=True)
            store.record_agent_event(run_id, 1, "coding", "sandbox_policy", {})
            self.assertTrue(store.finish_agent_run(run_id, "completed"))
            with self.assertRaisesRegex(ValueError, "only a running run"):
                store.mark_agent_run_unknown_outcome(run_id, confirmed_stopped=True)
            self.assertEqual(store.get_agent_run(run_id)["status"], "completed")

    def test_coding_run_creation_persists_identity_event_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "state.sqlite3")
            run_id = "d" * 32
            store.create_agent_run(run_id, "test task", "local", 2,
                                   initial_event=("coding", "run_started", {"run_id": run_id}))
            events = store.list_agent_events(run_id)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["sequence"], 0)
            self.assertEqual(events[0]["event_type"], "run_started")
            with self.assertRaises(sqlite3.IntegrityError):
                store.create_agent_run(run_id, "duplicate", "local", 2,
                                       initial_event=("coding", "run_started", {"run_id": run_id}))
            self.assertEqual(len(store.list_agent_events(run_id)), 1)

    def test_legacy_agent_run_status_constraint_migrates_without_losing_events(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "legacy.sqlite3"
            db = sqlite3.connect(database)
            db.executescript("""
                CREATE TABLE agent_runs (
                    id TEXT PRIMARY KEY, task TEXT NOT NULL,
                    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('local','remote')),
                    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','cancelled')),
                    max_steps INTEGER NOT NULL CHECK(max_steps BETWEEN 1 AND 4),
                    error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE agent_events (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0), actor TEXT NOT NULL,
                    event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(run_id,sequence));
                CREATE INDEX idx_agent_runs_created ON agent_runs(created_at DESC);
                CREATE INDEX idx_agent_events_run ON agent_events(run_id,sequence);
                INSERT INTO agent_runs VALUES('legacy-run','task',NULL,'local','running',2,'','2026-01-01','2026-01-01');
                INSERT INTO agent_events VALUES(1,'legacy-run',0,'coding','worktree_preparing','{}','2026-01-01');
            """)
            db.close()
            store = Store(database)
            self.assertEqual(store.get_agent_run("legacy-run")["status"], "running")
            self.assertEqual(len(store.list_agent_events("legacy-run")), 1)
            with store.connect() as migrated:
                schema = migrated.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='agent_runs'").fetchone()[0]
                self.assertIn("unknown_outcome", schema)
                self.assertEqual(migrated.execute("PRAGMA foreign_key_check").fetchall(), [])
            # Startup is idempotent after the one-time additive migration.
            Store(database)
            self.assertEqual(store.get_agent_run("legacy-run")["status"], "running")

    def test_code_recover_requires_cli_confirmation_and_reports_non_resuming_result(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state.sqlite3"
            store = Store(database)
            run_id = "c" * 32
            store.create_agent_run(run_id, "test task", "local", 2,
                                   initial_event=("coding", "run_started", {}))
            output = io.StringIO()
            env = {"AYU_DATABASE": str(database), "AYU_MODE": "local"}
            with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output):
                code = main(["code", "recover", run_id, "--confirm-process-stopped"])
            self.assertEqual(code, 0)
            response = json.loads(output.getvalue())
            self.assertEqual(response["status"], "unknown_outcome")
            self.assertEqual(store.get_agent_run(run_id)["status"], "unknown_outcome")
            self.assertIn(f"ayu code inspect {run_id}", response["next_steps"])

    def test_code_recover_cli_requires_explicit_confirmation_flag(self):
        with self.assertRaises(SystemExit) as error:
            main(["code", "recover", "a" * 32])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
