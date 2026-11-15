import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ayu.provider import ProviderError, VLLMProvider
from ayu.runtime import ChatRuntime, StreamingRuntime, StructuredOutputRuntime
from ayu.settings import Settings


def make_settings(**overrides):
    values = {
        "api_key": None, "base_url": "https://api.example.test/v1",
        "model": None, "database_path": Path("ayu-vllm-test.sqlite"),
        "local_backend": "vllm", "local_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "vllm_url": "http://127.0.0.1:8000", "local_timeout": 5,
    }
    values.update(overrides)
    return Settings(**values)


class FakeResponse:
    status = 200

    def __init__(self, payload=None, lines=None, content_type="application/json"):
        self.payload, self.lines, self.headers = payload, list(lines or []), {"Content-Type": content_type}

    def read(self, limit=-1):
        return json.dumps(self.payload).encode() if self.payload is not None else b""

    def readline(self, limit=-1):
        return self.lines.pop(0) if self.lines else b""

    def getheader(self, name):
        return self.headers.get(name)


class FakeConnection:
    def __init__(self, response):
        self.response, self.sock, self.calls = response, None, []

    def request(self, method, path, body=None, headers=None):
        self.calls.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        pass


def test_vllm_is_loopback_only_and_rejects_hostnames():
    with pytest.raises(ProviderError, match="loopback"):
        VLLMProvider(make_settings(vllm_url="http://192.168.1.4:8000"))._endpoint()
    with pytest.raises(ProviderError, match="numeric loopback"):
        VLLMProvider(make_settings(vllm_url="http://vllm:8000"))._endpoint()


def test_vllm_requires_exact_served_model_id():
    listing = {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"}]}
    connection = FakeConnection(FakeResponse(listing))
    with patch("ayu.provider.http.client.HTTPConnection", return_value=connection):
        assert VLLMProvider(make_settings()).validate() == "Qwen/Qwen2.5-7B-Instruct"
    with patch("ayu.provider.http.client.HTTPConnection", return_value=FakeConnection(FakeResponse(listing))):
        with pytest.raises(ProviderError, match="exactly match"):
            VLLMProvider(make_settings(local_model_id="qwen")).validate()


def test_vllm_chat_uses_selected_model_and_bounded_token_count():
    listing = {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
    completion = {"choices": [{"message": {"role": "assistant", "content": "local response"}}]}
    connections = [FakeConnection(FakeResponse(listing)), FakeConnection(FakeResponse(completion))]
    with patch("ayu.provider.http.client.HTTPConnection", side_effect=connections):
        answer = VLLMProvider(make_settings()).complete([{"role": "user", "content": "hello"}])
    assert answer == "local response"
    payload = json.loads(connections[1].calls[0][2])
    assert payload["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["max_tokens"] == 2048
    assert payload["stream"] is False


def test_vllm_structured_output_uses_json_schema_response_format():
    listing = {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
    completion = {"choices": [{"message": {"content": '{"ok":true}'}}]}
    connections = [FakeConnection(FakeResponse(listing)), FakeConnection(FakeResponse(completion))]
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    with patch("ayu.provider.http.client.HTTPConnection", side_effect=connections):
        answer = VLLMProvider(make_settings()).complete_structured(
            [{"role": "user", "content": "return json"}], schema, schema_name="result")
    assert answer == '{"ok":true}'
    assert json.loads(connections[1].calls[0][2])["response_format"]["json_schema"]["schema"] == schema


def test_vllm_streaming_parses_sse_and_conforms_to_runtime_protocols():
    listing = {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
    stream = FakeResponse(lines=[
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\\n',
        b'data: [DONE]\\n',
    ], content_type="text/event-stream")
    connections = [FakeConnection(FakeResponse(listing)), FakeConnection(stream)]
    with patch("ayu.provider.http.client.HTTPConnection", side_effect=connections):
        provider = VLLMProvider(make_settings())
        assert isinstance(provider, ChatRuntime)
        assert isinstance(provider, StreamingRuntime)
        assert isinstance(provider, StructuredOutputRuntime)
        assert "".join(provider.stream_complete([{"role": "user", "content": "hi"}])) == "hello world"
