from __future__ import annotations

import os
import subprocess
import socket
import tempfile
import unittest
import io
import contextlib
from pathlib import Path
from unittest.mock import patch

from ayu.cli import main
from ayu.sandbox import DockerSandboxRunner, SandboxError, SandboxPolicy
from ayu.store import Store


VALID_IMAGE = "registry.example.test/ayu/python-tests@sha256:" + "a" * 64


class SandboxPolicyTests(unittest.TestCase):
    def runner(self, **kwargs):
        return DockerSandboxRunner(image=VALID_IMAGE, **kwargs)

    def test_image_tags_are_rejected_without_docker_invocation(self):
        runner = DockerSandboxRunner(image="python:3.12-slim")
        with patch("ayu.sandbox.shutil.which") as which:
            with self.assertRaisesRegex(SandboxError, "immutable image digest"):
                runner.preflight()
        which.assert_not_called()

    def test_remote_docker_endpoints_are_rejected(self):
        runner = self.runner(docker_host="tcp://192.0.2.4:2376")
        with self.assertRaisesRegex(SandboxError, "Remote Docker daemon"):
            runner.preflight()

    def test_local_unix_socket_is_the_only_explicit_posix_endpoint(self):
        if os.name == "nt":
            self.skipTest("POSIX Unix socket endpoint policy")
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "docker.sock")
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(path)
                self.runner(docker_host="unix://" + path)._validate_local_host("unix://" + path)
                with self.assertRaisesRegex(SandboxError, "existing Unix socket"):
                    self.runner()._validate_local_host("unix:///run/nonexistent-ayu-docker.sock")
            finally:
                server.close()
        with self.assertRaises(SandboxError):
            self.runner(docker_host="unix://relative.sock")._validate_local_host("unix://relative.sock")

    def test_preflight_refuses_missing_docker_binary(self):
        runner = self.runner(docker_executable="ayu-docker-absent-test-binary")
        with patch("ayu.sandbox.shutil.which", return_value=None):
            with self.assertRaisesRegex(SandboxError, "Docker CLI is unavailable"):
                runner.preflight()

    def test_preflight_checks_local_daemon_and_image_without_pull(self):
        runner = self.runner()
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            class Result:
                returncode = 0
                stdout = ("27.0.1\n" if "info" in command else
                          '["PATH=/usr/local/bin","API_TOKEN=image-secret"]|sha256:' + "b" * 64 + "\n")
                stderr = ""
            return Result()

        with patch("ayu.sandbox.shutil.which", return_value="/usr/bin/docker"), \
             patch("ayu.sandbox.subprocess.run", side_effect=fake_run):
            result = runner.preflight()
        self.assertTrue(result["runtime_version"].startswith("27."))
        self.assertEqual([call[0][1:3] for call in calls], [["info", "--format"], ["image", "inspect"]])
        self.assertTrue(all(call[1]["timeout"] <= 8 for call in calls))
        self.assertTrue(all("DOCKER_CONFIG" in call[1]["env"] for call in calls))
        self.assertTrue(all("DOCKER_HOST" not in call[1]["env"] for call in calls))
        self.assertEqual(result["image_environment_keys"], ["PATH", "API_TOKEN"])
        self.assertNotIn("image-secret", str(result))

    def test_container_command_enforces_no_network_nonroot_and_resource_bounds(self):
        runner = self.runner()
        runner._docker_path = "/usr/bin/docker"
        with tempfile.TemporaryDirectory() as temp:
            command = runner._container_command(Path(temp), ["python", "-m", "pytest", "-q"], "ayu-test-unit",
                                                ["PATH", "API_TOKEN", "PYTHONPATH"])
        self.assertIn("--mount", command)
        mount = command[command.index("--mount") + 1]
        self.assertTrue(mount.endswith(",readonly"))
        self.assertIn("--env=TMPDIR=/tmp", command)
        self.assertIn("--env=PYTHONPYCACHEPREFIX=/tmp/pycache", command)
        self.assertIn("--env=API_TOKEN=", command)
        self.assertNotIn("--env=PATH=", command)
        self.assertIn("--workdir=/workspace", command)
        self.assertIn("--read-only", command)
        self.assertIn("--user", command)
        self.assertEqual(command[command.index("--user") + 1], "65532:65532")
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--security-opt=no-new-privileges:true", command)
        self.assertIn("--pids-limit=64", command)
        self.assertIn("--memory=536870912", command)
        self.assertIn("--memory-swap=536870912", command)
        self.assertIn("--cpus=1.0", command)
        self.assertIn("--rm", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("/var/run/docker.sock", command)
        self.assertEqual(SandboxPolicy(VALID_IMAGE).public_dict()["network"], "none")

    def test_fixed_suites_never_accept_model_selected_shell_commands(self):
        self.assertEqual(DockerSandboxRunner._suite_command("pytest", ["tests"]),
                         ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--", "tests"])
        with self.assertRaisesRegex(SandboxError, "Unsupported"):
            DockerSandboxRunner._suite_command("bash", ["-c", "id"])

    def test_path_traversal_and_symlink_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "tests").mkdir(); (root / "tests" / "test_ok.py").write_text("", encoding="utf-8")
            self.assertEqual(DockerSandboxRunner._checked_paths(root, "pytest", ["tests"]), ["tests"])
            for item in ("../outside", "/etc/passwd", ".git/config", "-h", "tests/\x00bad"):
                with self.subTest(item=item), self.assertRaises(SandboxError):
                    DockerSandboxRunner._checked_paths(root, "pytest", [item])
            target = root / "escape.py"
            outside = Path(temp) / "elsewhere.py"
            outside.write_text("", encoding="utf-8")
            try:
                target.symlink_to(outside)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(SandboxError, "Symbolic links"):
                DockerSandboxRunner._checked_paths(root, "pytest", ["escape.py"])

    def test_snapshot_omits_git_and_secret_files_but_keeps_source_and_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            (root / ".git").mkdir(); (root / ".git" / "config").write_text("private git metadata", encoding="utf-8")
            (root / ".ssh").mkdir(); (root / ".ssh" / "id_ed25519").write_text("private key", encoding="utf-8")
            (root / ".aws").mkdir(); (root / ".aws" / "credentials").write_text("cloud credentials", encoding="utf-8")
            import shutil
            shutil.rmtree(root / ".git")
            (root / ".git").write_text("gitdir: /host/repository/.git/worktrees/task", encoding="utf-8")
            (root / ".env").write_text("secret", encoding="utf-8")
            (root / ".env.local").write_text("secret", encoding="utf-8")
            (root / ".env.example").write_text("placeholder", encoding="utf-8")
            (root / "id_rsa").write_text("private key", encoding="utf-8")
            (root / "deployment.token").write_text("token", encoding="utf-8")
            (root / "module.py").write_text("print('ok')\n", encoding="utf-8")
            dest = Path(temp) / "snapshot"
            result = DockerSandboxRunner._snapshot(root, dest)
            self.assertTrue((dest / "module.py").is_file())
            self.assertTrue((dest / ".env.example").is_file())
            self.assertFalse((dest / ".git").exists())
            self.assertFalse((dest / ".ssh").exists())
            self.assertFalse((dest / ".aws").exists())
            self.assertFalse((dest / ".env").exists())
            self.assertFalse((dest / ".env.local").exists())
            self.assertFalse((dest / "id_rsa").exists())
            self.assertFalse((dest / "deployment.token").exists())
            self.assertGreaterEqual(result["paths_omitted"], 7)
            self.assertEqual(dest.stat().st_mode & 0o777, 0o755)
            self.assertEqual((dest / "module.py").stat().st_mode & 0o777, 0o444)

    def test_snapshot_omits_hardlinks_and_nested_mountpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            outside = Path(temp) / "host-data.txt"
            outside.write_text("sensitive host data", encoding="utf-8")
            try:
                os.link(outside, root / "ordinary-name.txt")
            except (OSError, NotImplementedError):
                self.skipTest("hard links are unavailable on this filesystem")
            mounted = root / "mounted"; mounted.mkdir()
            (mounted / "nested-secret.py").write_text("must not copy", encoding="utf-8")
            dest = Path(temp) / "snapshot"
            real_ismount = os.path.ismount
            is_mounted = lambda path: os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(mounted))
            with patch("ayu.sandbox.os.path.ismount", side_effect=lambda path: is_mounted(path) or real_ismount(path)):
                info = DockerSandboxRunner._snapshot(root, dest)
            self.assertFalse((dest / "ordinary-name.txt").exists())
            self.assertFalse((dest / "mounted").exists())
            self.assertGreaterEqual(info["paths_omitted"], 2)

    def test_snapshot_excludes_same_device_mountinfo_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            mounted = root / "same-device-bind"; mounted.mkdir()
            (mounted / "private.py").write_text("not copied", encoding="utf-8")
            mount_path = str(mounted)
            mountinfo = f"42 1 0:1 / {mount_path} rw - ext4 /dev/example rw\n"
            real_open = Path.open

            def fake_open(path, *args, **kwargs):
                if str(path).replace("\\", "/") == "/proc/self/mountinfo":
                    return io.StringIO(mountinfo)
                return real_open(path, *args, **kwargs)

            real_exists = Path.exists
            def fake_exists(path):
                if str(path).replace("\\", "/") == "/proc/self/mountinfo":
                    return True
                return real_exists(path)

            dest = Path(temp) / "snapshot"
            with patch.object(Path, "open", new=fake_open), patch.object(Path, "exists", new=fake_exists), \
                 patch("ayu.sandbox.os.path.ismount", return_value=False):
                result = DockerSandboxRunner._snapshot(root, dest)
            self.assertFalse((dest / "same-device-bind").exists())
            self.assertEqual(result["paths_omitted"], 1)

    def test_snapshot_entry_budget_includes_empty_and_filtered_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            (root / "a").mkdir(); (root / "b").mkdir(); (root / "c").mkdir()
            with patch("ayu.sandbox.MAX_SNAPSHOT_ENTRIES", 2):
                with self.assertRaisesRegex(SandboxError, "entry traversal bound"):
                    DockerSandboxRunner._snapshot(root, Path(temp) / "snapshot")

    def test_snapshot_stops_scanning_immediately_after_entry_limit(self):
        class Entry:
            def __init__(self, name): self.name = name
        class Scan:
            def __init__(self): self.visited = 0
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def __iter__(self):
                for index in range(100):
                    self.visited += 1
                    yield Entry(f"entry-{index}")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            scan = Scan()
            with patch("ayu.sandbox.MAX_SNAPSHOT_ENTRIES", 2), patch("ayu.sandbox.os.scandir", return_value=scan):
                with self.assertRaisesRegex(SandboxError, "entry traversal bound"):
                    DockerSandboxRunner._snapshot(root, Path(temp) / "snapshot")
            self.assertEqual(scan.visited, 3)

    def test_snapshot_enforces_preparation_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            with patch("ayu.sandbox.SNAPSHOT_TIMEOUT_SECONDS", -1):
                with self.assertRaisesRegex(SandboxError, "preparation deadline"):
                    DockerSandboxRunner._snapshot(root, Path(temp) / "snapshot")

    def test_snapshot_fails_closed_on_any_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir()
            outside = Path(temp) / "outside.py"; outside.write_text("not copied", encoding="utf-8")
            try:
                (root / "link.py").symlink_to(outside)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(SandboxError, "refuses symbolic links"):
                DockerSandboxRunner._snapshot(root, Path(temp) / "snapshot")

    def test_workspace_mount_paths_with_mount_delimiters_are_rejected(self):
        runner = self.runner(); runner._docker_path = "/usr/bin/docker"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "path,with,commas"
            path.mkdir()
            with self.assertRaisesRegex(SandboxError, "commas"):
                runner._container_command(path, ["python", "-m", "pytest"], "ayu-test-unit")

    def test_cli_sandbox_status_is_diagnostic_only(self):
        output = io.StringIO()
        class DiagnosticRunner:
            @classmethod
            def from_environment(cls):
                return cls()
            def status(self):
                return {"ready": False, "backend": "docker", "reason": "not configured"}
        with patch("ayu.sandbox.DockerSandboxRunner", DiagnosticRunner), contextlib.redirect_stdout(output):
            result = main(["sandbox", "status"])
        self.assertEqual(result, 2)
        self.assertIn('"ready": false', output.getvalue())

    def test_coding_help_does_not_offer_host_execution_consent(self):
        with self.assertRaises(SystemExit) as caught, patch("sys.stdout", new_callable=io.StringIO) as output:
            main(["code", "run", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("sandbox", output.getvalue().casefold())
        self.assertNotIn("--allow-host-tests", output.getvalue())

    def test_cleanup_refuses_non_ayu_container_names(self):
        runner = self.runner()
        with patch("ayu.sandbox.shutil.which") as which:
            with self.assertRaisesRegex(SandboxError, "not a valid Ayu"):
                runner.cleanup_container("user-database")
        which.assert_not_called()

    def test_cleanup_uses_only_the_explicit_generated_container_name(self):
        runner = self.runner()
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            class Result:
                returncode = 1 if "inspect" in command else 0
                stdout = b"27.0.1\n" if "info" in command else b""
                stderr = b"Error: No such object: container\n" if "inspect" in command else b""
            return Result()
        with patch("ayu.sandbox.shutil.which", return_value="/usr/bin/docker"), \
             patch("ayu.sandbox.subprocess.run", side_effect=fake_run):
            removed = runner.cleanup_container("ayu-test-1234567890abcdef-0123abcd")
        self.assertTrue(removed)
        self.assertEqual(calls[0][1:3], ["info", "--format"])
        self.assertEqual(calls[1], ["/usr/bin/docker", "rm", "--force", "ayu-test-1234567890abcdef-0123abcd"])
        self.assertEqual(calls[2], ["/usr/bin/docker", "inspect", "ayu-test-1234567890abcdef-0123abcd", "--format", "{{.Id}}"])

    def test_cli_cleanup_uses_only_persisted_container_start_events(self):
        with tempfile.TemporaryDirectory() as temp:
            run_id = "1234567890abcdef1234567890abcdef"
            store = Store(Path(temp) / "state.sqlite3")
            store.create_agent_run(run_id, "test task", "local", 2)
            name = "ayu-test-1234567890abcdef-0123abcd"
            store.record_agent_event(run_id, 0, "coding", "sandbox_container_starting",
                                     {"container_name": name, "backend": "docker"})
            called = []
            class CleanupRunner:
                def cleanup_container(self, container_name):
                    called.append(container_name)
                    return True
            output = io.StringIO()
            env = {"AYU_DATABASE": str(Path(temp) / "state.sqlite3")}
            with patch.dict(os.environ, env, clear=True), \
                 patch("ayu.sandbox.DockerSandboxRunner.from_environment", return_value=CleanupRunner()), \
                 contextlib.redirect_stdout(output):
                result = main(["sandbox", "cleanup", run_id])
            self.assertEqual(result, 0)
            self.assertEqual(called, [name])
            self.assertIn('"removed_or_absent": true', output.getvalue())

    def test_run_audits_container_intent_before_start_and_verifies_cleanup(self):
        runner = self.runner()
        runner._docker_path = "/usr/bin/docker"
        order = []
        class FakeProcess:
            stdout = __import__("io").BytesIO(b"tests passed")
            stderr = __import__("io").BytesIO(b"")
            returncode = 0
            def wait(self, timeout=None):
                return 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir(); (root / "tests").mkdir(); (root / "tests" / "test_x.py").write_text("", encoding="utf-8")
            with patch.object(runner, "preflight", return_value={**SandboxPolicy(VALID_IMAGE).public_dict(), "policy_version": "test"}), \
                 patch.object(runner, "_verify_cleanup", return_value=True), \
                 patch("ayu.sandbox.subprocess.Popen", side_effect=lambda *a, **kw: (order.append("Popen") or FakeProcess())):
                result = runner.run_tests(root, "pytest", ["tests"], 5, run_id="1234567890abcdef",
                                          on_start=lambda data: order.append(data["container_name"]))
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(order[0].split("-")[0:2], ["ayu", "test"])
        self.assertEqual(order[1], "Popen")

    def test_timeout_forces_removal_and_verifies_cleanup(self):
        runner = self.runner(); runner._docker_path = "/usr/bin/docker"
        class TimeoutProcess:
            stdout = io.BytesIO(b""); stderr = io.BytesIO(b""); pid = 2_147_483_647
            returncode = None; waits = 0
            def poll(self): return self.returncode
            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired("docker run", timeout)
                self.returncode = -15
                return self.returncode
        proc = TimeoutProcess()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir(); (root / "tests").mkdir()
            with patch.object(runner, "preflight", return_value=SandboxPolicy(VALID_IMAGE).public_dict()), \
                 patch.object(runner, "_remove_container", return_value=True) as remove, \
                 patch.object(runner, "_verify_cleanup", return_value=True) as verify, \
                 patch("ayu.sandbox.os.killpg", side_effect=ProcessLookupError), \
                 patch("ayu.sandbox.subprocess.Popen", return_value=proc):
                result = runner.run_tests(root, "pytest", ["tests"], 1, run_id="1234567890abcdef")
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["cleanup_confirmed"])
        remove.assert_called_once()
        verify.assert_called_once()

    def test_keyboard_interrupt_attempts_container_cleanup_then_propagates(self):
        runner = self.runner(); runner._docker_path = "/usr/bin/docker"
        class InterruptedProcess:
            stdout = io.BytesIO(b""); stderr = io.BytesIO(b""); pid = 2_147_483_647
            returncode = None; waits = 0
            def poll(self): return self.returncode
            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise KeyboardInterrupt
                self.returncode = -15
                return self.returncode
        proc = InterruptedProcess()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"; root.mkdir(); (root / "tests").mkdir()
            with patch.object(runner, "preflight", return_value=SandboxPolicy(VALID_IMAGE).public_dict()), \
                 patch.object(runner, "_remove_container", return_value=True) as remove, \
                 patch.object(runner, "_verify_cleanup", return_value=True) as verify, \
                 patch("ayu.sandbox.os.killpg", side_effect=ProcessLookupError), \
                 patch("ayu.sandbox.subprocess.Popen", return_value=proc):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_tests(root, "pytest", ["tests"], 1, run_id="1234567890abcdef")
        remove.assert_called_once()
        verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
