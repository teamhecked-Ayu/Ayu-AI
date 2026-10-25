from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import contextlib
import io

from ayu.cli import main
from ayu.provider import AnthropicProvider, ChatProvider, NewsAPIClient, ProviderError, build_remote_provider
from ayu.settings import Settings


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit=-1): return self.payload[:limit]


class ProviderIntegrationTests(unittest.TestCase):
    def test_news_cli_requires_per_request_network_consent(self):
        with tempfile.TemporaryDirectory() as temporary:
            env={"AYU_MODE":"local","AYU_DATABASE":str(Path(temporary)/"db.sqlite3"),"AYU_NEWSAPI_ENABLED":"true","NEWSAPI_API_KEY":"news-key-test"}
            errors=io.StringIO()
            with patch.dict(os.environ,env,clear=True),patch("ayu.provider.urllib.request.urlopen",side_effect=AssertionError("network must not be used")):
                with contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as caught:
                        main(["news","search","topic"])
            self.assertEqual(caught.exception.code,2)
            self.assertIn("--allow-network",errors.getvalue())

    def test_news_cli_uses_only_mocked_network_and_keeps_key_out_of_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            env={"AYU_MODE":"local","AYU_DATABASE":str(Path(temporary)/"db.sqlite3"),"AYU_NEWSAPI_ENABLED":"true","NEWSAPI_API_KEY":"news-key-test"}
            captured=[]
            payload={"status":"ok","totalResults":1,"articles":[{"source":{"name":"News"},"title":"Story","description":"Summary","url":"https://news.example/story","publishedAt":"2026-01-01T00:00:00Z"}]}
            def fake_urlopen(request,timeout):
                captured.append(request)
                return FakeResponse(payload)
            output=io.StringIO()
            with patch.dict(os.environ,env,clear=True),patch("ayu.provider.urllib.request.urlopen",side_effect=fake_urlopen):
                with contextlib.redirect_stdout(output):
                    code=main(["news","search","topic","--limit","1","--allow-network"])
            self.assertEqual(code,0)
            self.assertEqual(json.loads(output.getvalue())["articles"][0]["title"],"Story")
            self.assertEqual(captured[0].get_header("X-api-key"),"news-key-test")
            self.assertNotIn("news-key-test",captured[0].full_url)

    def test_provider_specific_environment_keys_are_selected_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            no_dotenv = Path(temporary) / "absent.env"
            database=str(Path(temporary)/"state.sqlite3")
            with patch.dict(os.environ, {"AYU_DATABASE":database,"AYU_PROVIDER": "openai", "OPENAI_API_KEY": "openai-test", "ANTHROPIC_API_KEY": "anthropic-test", "NEWSAPI_API_KEY": "news-test"}, clear=True):
                settings = Settings.load(no_dotenv)
            self.assertEqual(settings.api_key, "openai-test")
            self.assertEqual(settings.remote_provider, "openai")
            self.assertEqual(settings.newsapi_api_key, "news-test")
            with patch.dict(os.environ, {"AYU_DATABASE":database,"AYU_PROVIDER": "anthropic", "OPENAI_API_KEY": "openai-test", "ANTHROPIC_API_KEY": "anthropic-test"}, clear=True):
                settings = Settings.load(no_dotenv)
            self.assertEqual(settings.api_key, "anthropic-test")
            self.assertEqual(build_remote_provider(settings).__class__, AnthropicProvider)

    def test_openai_request_keeps_key_in_authorization_header(self):
        settings = Settings(api_key="openai-secret-test", base_url="https://api.openai.com/v1", model="test-model", database_path=Path("/tmp/ayu-test.sqlite3"), remote_provider="openai")
        response = {"choices": [{"message": {"content": "answer"}}]}
        captured = []
        def fake_urlopen(request, timeout):
            captured.append((request, timeout))
            return FakeResponse(response)
        with patch("ayu.provider.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(ChatProvider(settings).complete([{"role": "user", "content": "hello"}]), "answer")
        request, timeout = captured[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer openai-secret-test")
        self.assertNotIn("openai-secret-test", request.full_url)
        self.assertEqual(timeout, settings.request_timeout)

    def test_anthropic_adapter_uses_only_its_own_header_and_parses_text(self):
        settings = Settings(api_key="anthropic-secret-test", base_url="https://unused.invalid/v1", model="test-claude", database_path=Path("/tmp/ayu-test.sqlite3"), remote_provider="anthropic")
        captured = []
        payload = {"content": [{"type": "text", "text": "safe answer"}, {"type": "tool_use", "id": "ignored"}]}
        def fake_urlopen(request, timeout):
            captured.append(request)
            return FakeResponse(payload)
        with patch("ayu.provider.urllib.request.urlopen", side_effect=fake_urlopen):
            answer = AnthropicProvider(settings).complete([{"role": "system", "content": "be safe"}, {"role": "user", "content": "hello"}])
        self.assertEqual(answer, "safe answer")
        request = captured[0]
        self.assertEqual(request.get_header("X-api-key"), "anthropic-secret-test")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertNotIn("anthropic-secret-test", request.full_url)
        body = json.loads(request.data)
        self.assertEqual(body["system"], "be safe")
        self.assertNotIn("Authorization", request.headers)

    def test_newsapi_is_independent_gated_cached_and_filters_unsafe_urls(self):
        settings = Settings(api_key=None, base_url="https://unused.invalid/v1", model=None, database_path=Path("/tmp/ayu-test.sqlite3"), newsapi_api_key="news-secret-test", newsapi_enabled=True)
        NewsAPIClient._cache.clear()
        captured = []
        payload = {"status": "ok", "totalResults": 2, "articles": [
            {"source": {"name": "Example News"}, "title": "A title", "description": "A description", "url": "https://news.example/story", "publishedAt": "2026-01-01T10:00:00Z"},
            {"source": {"name": "Unsafe"}, "title": "bad", "description": "bad", "url": "javascript:alert(1)", "publishedAt": "2026-01-01T10:00:00Z"},
        ]}
        def fake_urlopen(request, timeout):
            captured.append((request, timeout))
            return FakeResponse(payload)
        with patch("ayu.provider.urllib.request.urlopen", side_effect=fake_urlopen):
            client = NewsAPIClient(settings)
            result = client.search("climate", from_date="2026-01-01", to_date="2026-01-02", page_size=5)
            again = client.search("climate", from_date="2026-01-01", to_date="2026-01-02", page_size=5)
        self.assertEqual(result, again)
        self.assertEqual(len(captured), 1)
        request, timeout = captured[0]
        self.assertEqual(request.get_header("X-api-key"), "news-secret-test")
        self.assertNotIn("news-secret-test", request.full_url)
        self.assertEqual(timeout, settings.request_timeout)
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(result["articles"][0]["source"], "Example News")
        self.assertNotIn("news-secret-test", json.dumps(result))

    def test_newsapi_requires_opt_in_key_and_valid_dates_without_network(self):
        disabled = Settings(api_key=None, base_url="https://unused.invalid/v1", model=None, database_path=Path("/tmp/ayu-test.sqlite3"), newsapi_api_key="key", newsapi_enabled=False)
        with patch("ayu.provider.urllib.request.urlopen", side_effect=AssertionError("network must not be called")):
            with self.assertRaisesRegex(ProviderError, "disabled"):
                NewsAPIClient(disabled).search("topic")
        enabled = Settings(api_key=None, base_url="https://unused.invalid/v1", model=None, database_path=Path("/tmp/ayu-test.sqlite3"), newsapi_api_key="key", newsapi_enabled=True)
        with patch("ayu.provider.urllib.request.urlopen", side_effect=AssertionError("network must not be called")):
            with self.assertRaisesRegex(ProviderError, "YYYY-MM-DD"):
                NewsAPIClient(enabled).search("topic", from_date="tomorrow")


if __name__ == "__main__":
    unittest.main()
