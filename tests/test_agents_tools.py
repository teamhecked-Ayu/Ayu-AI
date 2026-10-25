from __future__ import annotations

import json
import os
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.agents import AgentRunner
from ayu.cli import main
from ayu.store import Store
from ayu.tools import ToolError, ToolRegistry


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("unexpected provider call")
        return self.responses.pop(0)


class ToolRegistryTests(unittest.TestCase):
    def test_search_and_read_are_bounded_to_explicit_text_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("Ayu feature marker\nsecond line\n", encoding="utf-8")
            (root / "empty.txt").write_text("", encoding="utf-8")
            (root / ".env").write_text("API_KEY=never-read", encoding="utf-8")
            tools = ToolRegistry(root)
            matches = tools.invoke("search_text", {"query": "feature marker"})
            self.assertEqual(matches["matches"][0]["path"], "README.md")
            read = tools.invoke("read_file", {"path": "README.md", "start_line": 2, "end_line": 2})
            self.assertEqual(read["content"], "second line")
            self.assertEqual(tools.invoke("read_file", {"path":"empty.txt"})["content"],"")
            self.assertEqual({tool["side_effect"] for tool in tools.list_tools()}, {"read_only"})
            with self.assertRaises(ToolError):
                tools.invoke("read_file", {"path": ".env"})
            with self.assertRaises(ToolError):
                tools.invoke("exec", {"command": "echo unsafe"})
            with self.assertRaises(ToolError):
                tools.invoke("read_file", {"path": "README.md", "extra": True})

    def test_traversal_and_symlink_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            outside = Path(temporary) / "private.txt"
            root.mkdir()
            outside.write_text("must not be exposed", encoding="utf-8")
            tools = ToolRegistry(root)
            with self.assertRaises(ToolError):
                tools.invoke("read_file", {"path": "../private.txt"})
            try:
                (root / "link.txt").symlink_to(outside)
            except OSError as exc:
                if os.name == "nt": self.skipTest(f"Windows symlink creation unavailable: {exc}")
                raise
            with self.assertRaises(ToolError):
                tools.invoke("read_file", {"path": "link.txt"})


class AgentRunnerTests(unittest.TestCase):
    def test_remote_context_requires_explicit_consent_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); workspace=root/"workspace"; workspace.mkdir()
            database=root/"ayu.sqlite3"
            env={"AYU_DATABASE":str(database),"AYU_MODE":"remote"}
            stderr=io.StringIO()
            with patch.dict(os.environ,env,clear=True),contextlib.redirect_stderr(stderr):
                self.assertEqual(main(["agents","run","inspect local code","--workspace",str(workspace),"--mode","remote"]),2)
            self.assertIn("--allow-remote-context",stderr.getvalue())
            store=Store(database)
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0],0)

    def test_plan_tool_analysis_and_review_are_sequential_and_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notes.md").write_text("release readiness marker", encoding="utf-8")
            store = Store(root / "ayu.sqlite3")
            project_id = store.create_project("demo", str(root))
            provider = FakeProvider([
                json.dumps({"steps": [{"tool_id": "search_text", "arguments": {"query": "release readiness"}}]}),
                "The marker is present in notes.md at line 1.",
                json.dumps({"status": "supported", "unsupported_claims": [], "reason": "The report matches the single returned observation."}),
            ])
            runner = AgentRunner(provider, store, ToolRegistry(root), mode="local")
            result = runner.run("Find release readiness evidence", max_steps=2, project_id=project_id)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps_executed"], 1)
            self.assertEqual(len(provider.calls), 3)
            saved = store.get_agent_run(result["run_id"])
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["project_id"], project_id)
            events = store.list_agent_events(result["run_id"])
            self.assertEqual([event["actor"] for event in events], ["system", "planner", "researcher", "analyst", "reviewer"])

    def test_invalid_planned_path_fails_and_is_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            (Path(temporary) / "secret.txt").write_text("private", encoding="utf-8")
            store = Store(Path(temporary) / "state.sqlite3")
            provider = FakeProvider([
                json.dumps({"steps": [{"tool_id": "read_file", "arguments": {"path": "../secret.txt"}}]}),
            ])
            runner = AgentRunner(provider, store, ToolRegistry(root), mode="local")
            with self.assertRaisesRegex(Exception, "failed"):
                runner.run("Read the secret", max_steps=1)
            runs = []
            with store.connect() as db:
                runs = [dict(row) for row in db.execute("SELECT * FROM agent_runs")]
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "failed")
            self.assertTrue(any(event["event_type"] == "failure" for event in store.list_agent_events(runs[0]["id"])))
            self.assertEqual(len(provider.calls), 1)

    def test_plan_limits_and_exact_output_schema_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "state.sqlite3")
            provider = FakeProvider([json.dumps({"steps": [], "shell": "echo no"})])
            runner = AgentRunner(provider, store, ToolRegistry(root), mode="local")
            with self.assertRaisesRegex(Exception, "exactly"):
                runner.run("Do a bounded task")


if __name__ == "__main__":
    unittest.main()
