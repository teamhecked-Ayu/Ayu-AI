import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.settings import Settings
from ayu.provider import ChatProvider, ProviderError


class ConfigTests(unittest.TestCase):
    def test_loads_env_file_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("AYU_MODEL=from-file\nAYU_MAX_OUTPUT_TOKENS=500\n", encoding="utf-8")
            with patch.dict(os.environ, {"AYU_MODEL": "from-environment", "AYU_DATABASE": str(Path(tmp) / "db.sqlite")}, clear=True):
                settings = Settings.load(path)
            self.assertEqual(settings.model, "from-environment")
            self.assertEqual(settings.max_output_tokens, 500)

    def test_invalid_remote_url_is_deferred_until_remote_use(self):
        with patch.dict(os.environ, {"AYU_BASE_URL": "file:///tmp", "AYU_DATABASE": "/tmp/ayu-test.sqlite"}, clear=True):
            settings = Settings.load(Path("/nonexistent/.env"))
        self.assertEqual(settings.base_url, "file:///tmp")
        with self.assertRaisesRegex(ProviderError, "AYU_BASE_URL"):
            ChatProvider(settings.__class__(api_key="test",base_url=settings.base_url,model="m",database_path=settings.database_path)).complete([])


class ProviderTests(unittest.TestCase):
    def settings(self, **overrides):
        values = dict(api_key="test-key", base_url="https://api.example.test/v1", model="test-model",
                      database_path=Path("/tmp/ayu-test.sqlite"), request_timeout=1, max_output_tokens=77)
        values.update(overrides)
        from ayu.settings import Settings
        return Settings(**values)

    def test_parses_compatible_chat_response(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return json.dumps({"choices": [{"message": {"content": " Hello "}}]}).encode()
        with patch("urllib.request.urlopen", return_value=Response()) as open_url:
            answer = ChatProvider(self.settings()).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(answer, "Hello")
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    def test_missing_credentials_fails_clearly(self):
        with self.assertRaisesRegex(ProviderError, "AYU_API_KEY"):
            ChatProvider(self.settings(api_key=None)).complete([])

    def test_plaintext_http_is_rejected_before_sending_credentials(self):
        with self.assertRaisesRegex(ProviderError, "HTTPS"):
            ChatProvider(self.settings(base_url="http://api.example.test/v1")).complete([])


if __name__ == "__main__":
    unittest.main()
