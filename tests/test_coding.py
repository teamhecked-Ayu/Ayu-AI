from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.coding import CodeRunError, CodingRunner, apply_edits, run_approved_tests, sanitize_test_output
from ayu.cli import main
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

    def test_unsupported_test_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(CodeRunError, "Supported test suites"):
                run_approved_tests(root, "bash", [], 10, root / "home", allow_host_execution=True)

    def test_test_runner_is_fail_closed_without_direct_consent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(CodeRunError, "explicit allow_host_execution=True"):
                run_approved_tests(root, "compileall", [], 10, root / "home")

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
                CodingRunner(provider, store).run("Make add return the sum", root, suite="compileall", allow_host_tests=True)
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

    def test_json_parser_accepts_only_a_single_strict_json_fence(self):
        runner = CodingRunner(FakeProvider([]), Store(Path(tempfile.mkdtemp()) / "state.sqlite3"))
        parsed = runner._json('```json\n{"status":"supported","issues":[]}\n```', "Reviewer", {"status", "issues"})
        self.assertEqual(parsed["status"], "supported")
        with self.assertRaisesRegex(CodeRunError, "valid JSON"):
            runner._json('Here is the result: {"status":"supported","issues":[]}', "Reviewer", {"status", "issues"})

    def test_cli_requires_host_test_consent_before_provider_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"; root.mkdir()
            stderr = io.StringIO()
            env = {"AYU_MODE": "local", "AYU_DATABASE": str(Path(temp) / "state.sqlite3")}
            with patch.dict(os.environ, env, clear=True), patch("ayu.cli.build_local_provider") as build, contextlib.redirect_stderr(stderr):
                result = main(["code", "run", "make a change", "--workspace", str(root), "--test-suite", "compileall"])
            self.assertEqual(result, 2)
            self.assertIn("--allow-host-tests", stderr.getvalue())
            build.assert_not_called()

    def test_cli_requires_llama_server_for_structured_coding(self):
        with tempfile.TemporaryDirectory() as temp:
            stderr = io.StringIO()
            env = {"AYU_MODE": "local", "AYU_LOCAL_BACKEND": "llama-cli", "AYU_DATABASE": str(Path(temp) / "state.sqlite3")}
            argv = ["code", "run", "make a change", "--workspace", temp, "--test-suite", "compileall", "--allow-host-tests"]
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
            result = CodingRunner(provider, store).run(
                "Make add return the sum", root, suite="unittest", test_paths=["tests"], timeout=20, allow_host_tests=True
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
            self.assertIn("OK", test_evidence["stderr_excerpt"])
            subprocess.run(["git", "-C", str(worktree), "worktree", "remove", "--force", str(worktree)], check=True, capture_output=True)
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
            result = CodingRunner(provider, store).run(
                "Make add return the sum", root, suite="unittest", timeout=20, allow_host_tests=True
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["test"]["exit_code"], 0)
            self.assertEqual(len(provider.calls), 4)
            types = [event["event_type"] for event in store.list_agent_events(result["run_id"])]
            self.assertEqual(types.count("correction"), 1)
            worktree = Path(result["worktree"])
            subprocess.run(["git", "-C", str(worktree), "worktree", "remove", "--force", str(worktree)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "branch", "-D", result["branch"]], check=True, capture_output=True)

    def test_dirty_source_repository_is_refused_before_run_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {"calc.py": "x = 1\n"})
            (root / "calc.py").write_text("x = 2\n", encoding="utf-8")
            store = Store(Path(temp) / "state.sqlite3")
            runner = CodingRunner(FakeProvider([]), store)
            with self.assertRaisesRegex(CodeRunError, "dirty repository"):
                runner.run("change it", root, suite="compileall", allow_host_tests=True)
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0], 0)

    def test_git_filter_attributes_are_rejected_before_worktree_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            init_repo(root, {".gitattributes": "*.py filter=untrusted\n", "calc.py": "x = 1\n"})
            store = Store(Path(temp) / "state.sqlite3")
            provider = FakeProvider([])
            runner = CodingRunner(provider, store)
            with self.assertRaisesRegex(CodeRunError, "content filter"):
                runner.run("make a code change", root, suite="compileall", allow_host_tests=True)
            self.assertEqual(provider.calls, [])
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
