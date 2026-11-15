"""Optional real Docker boundary tests; never infer isolation from mocks.

Set AYU_TEST_SANDBOX_IMAGE to a locally available digest-pinned image with
Python and unittest installed. These checks exercise a real Docker Engine,
but do not prove container escape resistance or verify every platform/runtime.
"""
from __future__ import annotations

import os
import socket
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from ayu.sandbox import DockerSandboxRunner


TEST_IMAGE = os.environ.get("AYU_TEST_SANDBOX_IMAGE", "").strip()
IMAGE_CONFIGURED = bool(TEST_IMAGE)


@unittest.skipUnless(
    IMAGE_CONFIGURED,
    "set AYU_TEST_SANDBOX_IMAGE to a local digest-pinned Python image for real Docker boundary checks",
)
class RealDockerBoundaryTests(unittest.TestCase):
    def _run_real_container(self, root: Path, *, timeout: int, run_id: str) -> dict:
        environment = {
            "AYU_SANDBOX_IMAGE": TEST_IMAGE,
            "AYU_TEST_HOST_SECRET": "host-only-value",
        }
        with patch.dict(os.environ, environment, clear=False):
            runner = DockerSandboxRunner.from_environment()
            return runner.run_tests(root, "unittest", ["tests"], timeout, run_id=run_id)

    def test_real_container_enforces_security_and_resource_boundaries(self):
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
    def test_non_root_and_privilege_restrictions(self):
        self.assertEqual(os.geteuid(), 65532)
        status = {{}}
        for line in Path("/proc/self/status").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        self.assertEqual(status.get("NoNewPrivs"), "1")
        self.assertEqual(status.get("CapEff"), "0000000000000000")

    def test_root_and_tmp_mount_options(self):
        readonly = getattr(os, "ST_RDONLY", 1)
        noexec = getattr(os, "ST_NOEXEC", 8)
        self.assertTrue(os.statvfs("/").f_flag & readonly)
        self.assertTrue(os.statvfs("/tmp").f_flag & noexec)
        with self.assertRaises(OSError):
            Path("/workspace/should-not-write.txt").write_text("no")
        probe = Path("/tmp/ayu-runtime-write-probe")
        probe.write_text("temporary")
        self.assertEqual(probe.read_text(), "temporary")
        probe.unlink()

    def test_host_environment_and_files_are_not_forwarded(self):
        self.assertIsNone(os.environ.get("AYU_TEST_HOST_SECRET"))
        self.assertFalse(Path({str(outside_secret)!r}).exists())

    def test_network_namespace_is_isolated(self):
        names = {{name for _, name in socket.if_nameindex()}}
        self.assertLessEqual(names, {{"lo"}})
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", {port}), timeout=1.0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.75)
            self.assertNotEqual(client.connect_ex(("1.1.1.1", 443)), 0)

    def test_cgroup_limits_are_applied(self):
        def read(path):
            try:
                return Path(path).read_text().strip()
            except OSError:
                return None

        if read("/sys/fs/cgroup/cgroup.controllers") is not None:
            pids = read("/sys/fs/cgroup/pids.max")
            memory = read("/sys/fs/cgroup/memory.max")
            cpu = read("/sys/fs/cgroup/cpu.max")
            self.assertEqual(pids, "64")
            self.assertEqual(int(memory), 536870912)
            quota, period = map(int, cpu.split())
        else:
            pids = read("/sys/fs/cgroup/pids/pids.max")
            memory = read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            quota = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            period = read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            self.assertEqual(pids, "64")
            self.assertIsNotNone(memory)
            self.assertLessEqual(int(memory), 536870912)
            self.assertIsNotNone(quota)
            self.assertIsNotNone(period)
            quota, period = int(quota), int(period)
        self.assertGreater(quota, 0)
        self.assertLessEqual(quota, period)
'''
                (tests / "test_boundary.py").write_text(source, encoding="utf-8")
                result = self._run_real_container(
                    root, timeout=30, run_id=uuid.uuid4().hex
                )
                self.assertFalse(result["infrastructure_error"], result["stderr"])
                self.assertFalse(result["timed_out"])
                self.assertEqual(result["exit_code"], 0, result["stdout"] + result["stderr"])
                self.assertTrue(result["cleanup_confirmed"])
                self.assertEqual(result["backend"], "docker")
                self.assertEqual(result["policy_version"], "ayu-docker-v1")
            finally:
                listener.close()

    def test_timed_out_container_is_removed_and_verified(self):
        with tempfile.TemporaryDirectory(prefix="ayu-sandbox-timeout-test-") as temporary:
            root = Path(temporary) / "repo"
            tests = root / "tests"
            tests.mkdir(parents=True)
            (tests / "test_timeout.py").write_text(
                "import time, unittest\n"
                "class SlowTest(unittest.TestCase):\n"
                "    def test_wait(self):\n"
                "        time.sleep(30)\n",
                encoding="utf-8",
            )
            result = self._run_real_container(
                root, timeout=3, run_id=uuid.uuid4().hex
            )
            self.assertTrue(result["timed_out"])
            self.assertTrue(result["cleanup_confirmed"])
            self.assertEqual(result["policy_version"], "ayu-docker-v1")


if __name__ == "__main__":
    unittest.main()
