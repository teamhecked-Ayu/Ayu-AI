"""llama-server contract tests; the fake HTTP server is not an LLM test."""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.cli import main
from ayu.provider import LlamaServerProvider, ProviderError
from ayu.settings import Settings


class FakeLlamaServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address):
        super().__init__(address, FakeLlamaHandler)
        self.events = []
        self.model_ids = ["test-local-model"]
        self.context_size = 4096
        self.model_path = ""
        self.build_info = "fake-test-server"
        self.input_tokens = 64
        self.health_status = 200
        self.health_body = {"status": "ok"}
        self.answer = "local test response"
        self.stream_chunks = ["local ", "stream ", "response"]
        self.stream_delay_seconds = 0.0
        self.stream_invalid = False
        self.token_request = None
        self.chat_request = None
        self.authorization_headers = []


class FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, status: int, body: dict | str):
        raw = (json.dumps(body) if isinstance(body, dict) else body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        state = self.server
        state.events.append(("GET", self.path))
        if self.path == "/health":
            self._send(state.health_status, state.health_body)
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": model_id} for model_id in state.model_ids]})
        elif self.path == "/props":
            self._send(200, {"model_path": state.model_path,
                             "build_info": state.build_info,
                             "default_generation_settings": {"n_ctx": state.context_size}})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        state = self.server
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        state.events.append(("POST", self.path))
        state.authorization_headers.append(self.headers.get("Authorization"))
        if self.path == "/v1/chat/completions/input_tokens":
            state.token_request = request
            self._send(200, {"input_tokens": state.input_tokens})
        elif self.path == "/v1/chat/completions":
            state.chat_request = request
            if request.get("stream"):
                if state.stream_invalid:
                    events = [b"data: not-json\n\n", b"data: [DONE]\n\n"]
                else:
                    events = [b"data: " + json.dumps({"choices": [{"delta": {"content": part}}]}).encode() + b"\n\n"
                              for part in state.stream_chunks]
                    events.append(b"data: [DONE]\n\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Content-Length", str(sum(map(len, events))))
                self.end_headers()
                try:
                    if state.stream_delay_seconds and events:
                        self.wfile.write(events[0])
                        self.wfile.flush()
                        time.sleep(state.stream_delay_seconds)
                        events = events[1:]
                    for event in events:
                        self.wfile.write(event)
                    self.wfile.flush()
                except OSError:
                    pass  # The client may deliberately close on timeout/cancellation.
            else:
                self._send(200, {"choices": [{"message": {"content": state.answer}}]})
        else:
            self._send(404, {"error": "not found"})


class LlamaServerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model = Path(self.temp.name) / "placeholder.gguf"
        # This deliberately is not a valid model. These tests exercise HTTP
        # protocol and validation only, never claim model loading or inference.
        self.model.write_bytes(b"test fixture only; not GGUF weights")
        self.server = FakeLlamaServer(("127.0.0.1", 0))
        self.server.model_path = str(self.model.resolve())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def settings(self, **overrides):
        values = dict(
            api_key=None,
            base_url="https://unused.invalid/v1",
            model=None,
            database_path=Path(self.temp.name) / "ayu.sqlite3",
            local_model_path=self.model,
            local_backend="llama-server",
            llama_server_url=self.url,
            local_model_id=None,
            local_timeout=2.0,
            max_output_tokens=128,
            local_max_input_chars=4096,
        )
        values.update(overrides)
        return Settings(**values)

    def test_chat_messages_are_preserved_and_template_token_budget_is_checked(self):
        messages = [
            {"role": "system", "content": "Use the model's chat template."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "Please calculate."},
        ]
        answer = LlamaServerProvider(self.settings()).complete(messages)
        self.assertEqual(answer, "local test response")
        self.assertEqual(self.server.token_request["messages"], messages)
        self.assertEqual(self.server.chat_request["messages"], messages)
        self.assertEqual(self.server.chat_request["model"], "test-local-model")
        self.assertEqual(self.server.chat_request["max_tokens"], 128)
        self.assertFalse(self.server.chat_request["stream"])
        self.assertEqual(self.server.authorization_headers, [None, None])
        self.assertIn(("POST", "/v1/chat/completions/input_tokens"), self.server.events)
        self.assertIn(("POST", "/v1/chat/completions"), self.server.events)

    def test_structured_completion_sends_strict_json_schema_after_token_preflight(self):
        answer = '{"summary":"Update add","edits":[]}'
        self.server.answer = answer
        schema = {"type": "object", "properties": {"summary": {"type": "string"}, "edits": {"type": "array"}}, "required": ["summary", "edits"], "additionalProperties": False}
        provider = LlamaServerProvider(self.settings())
        messages = [{"role": "user", "content": "Return structured JSON."}]
        self.assertEqual(provider.complete_structured(messages, schema, schema_name="ayu_test"), answer)
        self.assertEqual(self.server.token_request["messages"], messages)
        self.assertNotIn("response_format", self.server.token_request)
        self.assertEqual(self.server.chat_request["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "ayu_test", "strict": True, "schema": schema},
        })
        self.assertEqual(self.server.authorization_headers, [None, None])

    def test_sse_streaming_yields_text_chunks_after_token_preflight(self):
        provider = LlamaServerProvider(self.settings())
        pieces = list(provider.stream_complete([{"role": "user", "content": "stream a test"}]))
        self.assertEqual("".join(pieces), "local stream response")
        self.assertFalse(self.server.token_request["stream"])
        self.assertTrue(self.server.chat_request["stream"])
        self.assertEqual(self.server.authorization_headers, [None, None])

    def test_malformed_sse_event_is_suppressed_as_provider_error(self):
        self.server.stream_invalid = True
        with self.assertRaisesRegex(ProviderError, "malformed streaming data"):
            list(LlamaServerProvider(self.settings()).stream_complete([{"role": "user", "content": "test"}]))

    def test_stream_deadline_times_out_and_generator_close_is_safe(self):
        self.server.stream_delay_seconds = 1.5
        stream = LlamaServerProvider(self.settings(local_timeout=1.0)).stream_complete(
            [{"role": "user", "content": "test"}])
        self.assertEqual(next(stream), "local ")
        with self.assertRaisesRegex(ProviderError, "stream exceeded 1 seconds"):
            next(stream)
        stream.close()

    def test_context_overflow_is_rejected_before_generation(self):
        self.server.input_tokens = 4000
        with self.assertRaisesRegex(ProviderError, "4000 input tokens.*128 output tokens.*4096-token context"):
            LlamaServerProvider(self.settings()).complete([{"role": "user", "content": "test"}])
        self.assertNotIn(("POST", "/v1/chat/completions"), self.server.events)

    def test_input_character_limit_rejects_before_network_access(self):
        provider = LlamaServerProvider(self.settings(local_max_input_chars=3))
        with self.assertRaisesRegex(ProviderError, "AYU_LOCAL_MAX_INPUT_CHARS"):
            provider.complete([{"role": "user", "content": "four"}])
        self.assertEqual(self.server.events, [])

    def test_malformed_message_types_fail_as_provider_errors(self):
        provider = LlamaServerProvider(self.settings())
        for messages in ([{"role": [], "content": "bad"}], [{"role": "user", "content": 3}], [{"role": "user", "content": "x", "tools": []}]):
            with self.subTest(messages=messages), self.assertRaises(ProviderError):
                provider.complete(messages)
        self.assertEqual(self.server.events, [])

    def test_invalid_generation_budget_fails_before_network_access(self):
        with self.assertRaisesRegex(ProviderError, "AYU_MAX_OUTPUT_TOKENS"):
            LlamaServerProvider(self.settings(max_output_tokens="invalid")).complete(
                [{"role": "user", "content": "test"}])
        self.assertEqual(self.server.events, [])

    def test_server_is_restricted_to_numeric_loopback_without_dns_or_proxy(self):
        for url in ("http://localhost:9931", "http://192.0.2.10:9931", "https://127.0.0.1:9931", "http://127.0.0.1:9931/path", "http://127.0.0.1:0", "http://[::1%25lo]:9931"):
            with self.subTest(url=url), self.assertRaises(ProviderError):
                LlamaServerProvider(self.settings(llama_server_url=url)).validate()
        self.assertEqual(self.server.events, [])

    def test_model_id_is_required_when_server_lists_multiple_models(self):
        self.server.model_ids = ["one", "two"]
        with self.assertRaisesRegex(ProviderError, "multiple models"):
            LlamaServerProvider(self.settings()).validate()
        self.assertEqual(LlamaServerProvider(self.settings(local_model_id="two")).validate(), "two")
        with self.assertRaisesRegex(ProviderError, "did not match exactly one"):
            LlamaServerProvider(self.settings(local_model_id="missing")).validate()

    def test_server_loaded_model_must_match_configured_model_file(self):
        different = Path(self.temp.name) / "different.gguf"
        different.write_bytes(b"another fixture; not GGUF weights")
        self.server.model_path = str(different.resolve())
        with self.assertRaisesRegex(ProviderError, "does not match AYU_LOCAL_MODEL"):
            LlamaServerProvider(self.settings()).validate()
        self.assertNotIn(("POST", "/v1/chat/completions"), self.server.events)

    def test_nonready_server_error_does_not_echo_body(self):
        self.server.health_status = 503
        self.server.health_body = {"error": "private prompt-like response body"}
        with self.assertRaisesRegex(ProviderError, "HTTP 503") as caught:
            LlamaServerProvider(self.settings()).validate()
        self.assertNotIn("private prompt-like", str(caught.exception))

    def test_cli_distinguishes_preflight_from_generation_request(self):
        env = {
            "AYU_DATABASE": str(Path(self.temp.name) / "cli.sqlite3"),
            "AYU_MODE": "local",
            "AYU_LOCAL_BACKEND": "llama-server",
            "AYU_LOCAL_MODEL": str(self.model),
            "AYU_LLAMA_SERVER_URL": self.url,
            "AYU_MAX_OUTPUT_TOKENS": "64",
        }
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output):
            self.assertEqual(main(["models", "verify"]), 0)
        self.assertIn("this did not run model generation", output.getvalue())
        self.assertNotIn(("POST", "/v1/chat/completions"), self.server.events)

    def test_cli_infer_flag_sends_a_bounded_local_generation_request(self):
        env = {
            "AYU_DATABASE": str(Path(self.temp.name) / "cli-infer.sqlite3"),
            "AYU_MODE": "local",
            "AYU_LOCAL_BACKEND": "llama-server",
            "AYU_LOCAL_MODEL": str(self.model),
            "AYU_LLAMA_SERVER_URL": self.url,
            "AYU_MAX_OUTPUT_TOKENS": "64",
        }
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output):
            self.assertEqual(main(["models", "verify", "--infer"]), 0)
        self.assertIn("Local inference request succeeded", output.getvalue())
        self.assertIn("local test response", output.getvalue())
        self.assertIn(("POST", "/v1/chat/completions"), self.server.events)

    def test_cli_streams_and_persists_only_the_complete_answer(self):
        db = Path(self.temp.name) / "stream-chat.sqlite3"
        env = {"AYU_DATABASE": str(db), "AYU_MODE": "local", "AYU_LOCAL_BACKEND": "llama-server",
               "AYU_LOCAL_MODEL": str(self.model), "AYU_LLAMA_SERVER_URL": self.url,
               "AYU_MAX_OUTPUT_TOKENS": "64"}
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output):
            self.assertEqual(main(["chat", "--mode", "local", "--once", "say something"]), 0)
        self.assertIn("Ayu: local stream response", output.getvalue())
        import sqlite3
        with contextlib.closing(sqlite3.connect(db)) as connection:
            saved = connection.execute("SELECT content FROM messages WHERE role='assistant'").fetchone()[0]
        self.assertEqual(saved, "local stream response")

    def test_cli_stream_timeout_does_not_persist_partial_assistant_answer(self):
        self.server.stream_delay_seconds = 1.5
        db = Path(self.temp.name) / "partial-stream.sqlite3"
        env = {"AYU_DATABASE": str(db), "AYU_MODE": "local", "AYU_LOCAL_BACKEND": "llama-server",
               "AYU_LOCAL_MODEL": str(self.model), "AYU_LLAMA_SERVER_URL": self.url,
               "AYU_LOCAL_TIMEOUT": "1", "AYU_MAX_OUTPUT_TOKENS": "64"}
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(main(["chat", "--mode", "local", "--once", "say something"]), 1)
        self.assertIn("Ayu error", errors.getvalue())
        self.assertIn("local ", output.getvalue())  # Partial text may have been displayed, but is not durable history.
        import sqlite3
        with contextlib.closing(sqlite3.connect(db)) as connection:
            roles = connection.execute("SELECT role FROM messages ORDER BY id").fetchall()
        self.assertEqual(roles, [("user",)])


_REAL_URL = os.environ.get("AYU_TEST_LLAMA_SERVER_URL")
_REAL_MODEL = os.environ.get("AYU_TEST_LLAMA_MODEL")


@unittest.skipUnless(_REAL_URL and _REAL_MODEL,
                     "set AYU_TEST_LLAMA_SERVER_URL and AYU_TEST_LLAMA_MODEL to run against an explicitly configured real llama-server and GGUF")
class RealLlamaServerIntegrationTests(unittest.TestCase):
    def test_real_configured_model_generates_nonempty_response(self):
        model = Path(_REAL_MODEL).expanduser()
        settings = Settings(
            api_key=None,
            base_url="https://unused.invalid/v1",
            model=None,
            database_path=Path(tempfile.gettempdir()) / "ayu-real-llama-test.sqlite3",
            local_model_path=model,
            local_backend="llama-server",
            llama_server_url=_REAL_URL,
            local_model_id=os.environ.get("AYU_TEST_LLAMA_MODEL_ID") or None,
            local_timeout=180.0,
            max_output_tokens=96,
            local_max_input_chars=4096,
        )
        provider = LlamaServerProvider(settings)
        model_id = provider.validate()
        answer = provider.complete([
            {"role": "system", "content": "Answer briefly and do not include hidden reasoning."},
            {"role": "user", "content": "Reply with a short greeting."},
        ])
        self.assertTrue(answer.strip())
        streamed = "".join(provider.stream_complete([
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Say hello in a few words."},
        ]))
        self.assertTrue(streamed.strip())
        print(json.dumps({"integration": "real llama-server request", "model_id": model_id,
                          "model_path": str(model.resolve()), "nonempty_response": True,
                          "nonempty_stream": True}, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
