from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from ayu.config import default_config, load_config, save_config


class LegacyConfigTests(unittest.TestCase):
    def test_default_nested_configuration_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            config = default_config()
            save_config(config, path)
            self.assertEqual(load_config(path), config)

    def test_malformed_toml_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.toml"
            path.write_text("[unterminated", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Could not read TOML"):
                load_config(path)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits do not map consistently on Windows")
    def test_saved_config_is_owner_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            save_config({"mode": "basic", "offline": True}, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_legacy_status_exports_remain_available(self):
        from ayu.config import get_status, set_mode, show_config, validate_config
        self.assertTrue(callable(get_status))
        self.assertTrue(callable(set_mode))
        self.assertTrue(callable(show_config))
        self.assertTrue(callable(validate_config))


if __name__ == "__main__":
    unittest.main()
