"""Optional real Docker boundary tests; never infer isolation from mocks.

Set AYU_TEST_SANDBOX_IMAGE to a locally available digest-pinned image with
Python and unittest installed. These checks do not prove container escape
resistance or verify all platforms/runtime configurations.
"""
from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.sandbox import DockerSandboxRunner, SandboxError


TEST_IMAGE = os.environ.get("AYU_TEST_SANDBOX_IMAGE", "").strip()
IMAGE_CONFIGURED = bool(TEST_IMAGE)


@unittest.skipUnless(IMAGE_CONFIGURED, "set AYU_TEST_SANDBOX_IMAGE to a local digest-pinned Python test image for real Docker boundary checks")
class RealDockerBoundaryTests(unittest.TestCase):
    def test_container_cannot_reach_host_network_or_host_marker_and_workspace_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix="ayu-sandbox-real-test-") as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            outside_secret = Path(temporary) / "host-only-secret.txt"
            outside_secret.write_text("host marker; not part of the task snapshot", encoding="utf-8")
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            try:
                tests = root / "tests"
                tests.mkdir()
                source = f'''import os, socket, unittest
from pathlib import Path

class BoundaryTests(unittest.TestCase):
    def test_non_root(self):
        self.assertNotEqual(os.geteuid(), 0)
    def test_host_environment_is_not_forwarded(self):
        self.assertIsNone(os.environ.get("AYU_TEST_HOST_SECRET"))
    def test_only_loopback_network_interface_is_present(self):
        names = {{name for _, name in socket.if_nameindex()}}
        self.assertLessEqual(names, {{"lo"}})
    def test_host_loopback_service_is_unreachable(self):
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", {port}), timeout=1.0)
    def test_host_marker_is_not_mounted(self):
        self.assertFalse(Path({str(outside_secret)!r}).exists())
    def test_task_snapshot_is_read_only(self):
        with self.assertRaises(OSError):
            Path("/workspace/should-not-write.txt").write_text("no")
'''
                (tests / "test_boundary.py").write_text(source, encoding="utf-8")
                environment = {"AYU_SANDBOX_IMAGE": TEST_IMAGE, "AYU_TEST_HOST_SECRET": "host-only-value"}
                with patch.dict(os.environ, environment, clear=False):
                    runner = DockerSandboxRunner.from_environment()
                    result = runner.run_tests(root, "unittest", ["tests"], 30, run_id="real-boundary-check")
                self.assertFalse(result["infrastructure_error"], result["stderr"])
                self.assertFalse(result["timed_out"])
                self.assertEqual(result["exit_code"], 0, result["stdout"] + result["stderr"])
                self.assertTrue(result["cleanup_confirmed"])
                self.assertEqual(result["backend"], "docker")
            finally:
                listener.close()


if __name__ == "__main__":
    unittest.main()
