import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.diagnostics import collect_platform_diagnostics
from ayu.settings import Settings


class PlatformDiagnosticsTests(unittest.TestCase):
    def settings(self, root):
        return Settings(
            api_key="do-not-print-this-secret",
            base_url="https://unused.invalid/v1",
            model="private-model-name",
            database_path=Path(root) / "ayu.sqlite3",
            mode="local",
            local_backend="llama-server",
            local_model_id="private-model-id",
        )

    def test_report_is_offline_and_redacts_credentials_and_paths(self):
        with tempfile.TemporaryDirectory() as root:
            report = collect_platform_diagnostics(self.settings(root))
            rendered = str(report)
            self.assertEqual(report["schema_version"], "ayu-platform-diagnostics-v1")
            self.assertEqual(report["privacy"]["network_requests"], 0)
            self.assertFalse(report["privacy"]["secrets_included"])
            self.assertFalse(report["privacy"]["absolute_paths_included"])
            self.assertNotIn("do-not-print-this-secret", rendered)
            self.assertNotIn("private-model-name", rendered)
            self.assertNotIn(str(root), rendered)
            self.assertFalse(report["runtime"]["server_contacted"])
            self.assertEqual(report["runtime"]["local_backend"], "llama-server")

    def test_detects_termux_without_exposing_environment_values(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"ANDROID_ROOT": "/system", "PREFIX": "/data/data/com.termux/files/usr", "TERMUX_VERSION": "test"}, clear=True):
                with patch("ayu.diagnostics.platform.system", return_value="Linux"):
                    report = collect_platform_diagnostics(self.settings(root))
            self.assertEqual(report["platform"]["profile"], "android-termux")
            self.assertNotIn("/data/data", str(report))

    def test_missing_database_directory_is_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            settings = self.settings(Path(root) / "missing")
            report = collect_platform_diagnostics(settings)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(next(x for x in report["checks"] if x["id"] == "database-directory")["status"], "fail")

    def test_cli_json_output(self):
        from ayu.cli import main
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as root:
            output = io.StringIO()
            with patch.dict(os.environ, {"AYU_DATABASE": str(Path(root) / "db.sqlite3"), "AYU_MODE": "basic"}, clear=True):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(["doctor", "--json"]), 0)
            self.assertIn('"schema_version": "ayu-platform-diagnostics-v1"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
