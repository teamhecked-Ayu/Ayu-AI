import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ayu.provider import OllamaProvider, ProviderError
from ayu.settings import Settings


def make_settings(**overrides):
    values = {
        "api_key": None,
        "base_url": "https://api.example.test/v1",
        "model": None,
        "database_path": Path("ayu-test.sqlite"),
        "local_backend": "ollama",
        "local_model_id": "qwen2.5:7b",
        "ollama_url": "http://127.0.0.1:11434",
        "request_timeout": 1,
        "local_timeout": 5,
    }
    values.update(overrides)
    return Settings(**values)


class FakeResponse:
    status = 200

    def __init__(self, payload=None, lines=None):
        self.payload = payload
        self.lines = list(lines or [])

    def read(self, limit=-1):
        return json.dumps(self.payload).encode() if self.payload is not None else b""

    def readline(self, limit=-1):
        return self.lines.pop(0) if self.lines else b""


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.sock = None
        self.calls = []

    def request(self, method, path, body=None, headers=None):
        self.calls.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        pass


def test_ollama_endpoint_rejects_non_loopback_and_hostnames():
    provider = OllamaProvider(make_settings(ollama_url="http://192.168.1.5:11434"))
    with pytest.raises(ProviderError, match="loopback"):
        provider._endpoint()
    provider = OllamaProvider(make_settings(ollama_url="http://ollama:11434"))
    with pytest.raises(ProviderError, match="numeric loopback"):
        provider._endpoint()


def test_ollama_model_selection_requires_exact_id_when_multiple_models_are_installed():
    provider = OllamaProvider(make_settings())
    listing = {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.2:3b"}]}
    connection = FakeConnection(FakeResponse(listing))
    with patch("ayu.provider.http.client.HTTPConnection", return_value=connection):
        assert provider.validate() == "qwen2.5:7b"
    provider = OllamaProvider(make_settings(local_model_id="qwen"))
    connection = FakeConnection(FakeResponse(listing))
    with patch("ayu.provider.http.client.HTTPConnection", return_value=connection):
        with pytest.raises(ProviderError, match="exactly match"):
            provider.validate()


def test_ollama_chat_uses_only_selected_local_model_and_bounded_generation():
    listing = {"models": [{"name": "qwen2.5:7b", "digest": "sha256:abc"}]}
    completion = {"message": {"role": "assistant", "content": "local answer"}, "done": True}
    connections = [
        FakeConnection(FakeResponse(listing)),
        FakeConnection(FakeResponse(completion)),
    ]
    with patch("ayu.provider.http.client.HTTPConnection", side_effect=connections):
        answer = OllamaProvider(make_settings()).complete([{"role": "user", "content": "hello"}])
    assert answer == "local answer"
    request = json.loads(connections[1].calls[0][2])
    assert request["model"] == "qwen2.5:7b"
    assert request["stream"] is False
    assert request["options"]["num_predict"] == 2048


def test_ollama_structured_output_sends_schema_to_backend():
    listing = {"models": [{"name": "qwen2.5:7b"}]}
    completion = {"message": {"role": "assistant", "content": '{"ok":true}'}, "done": True}
    connections = [
        FakeConnection(FakeResponse(listing)),
        FakeConnection(FakeResponse(completion)),
    ]
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    with patch("ayu.provider.http.client.HTTPConnection", side_effect=connections):
        answer = OllamaProvider(make_settings()).complete_structured(
            [{"role": "user", "content": "return json"}], schema, schema_name="result"
        )
    assert answer == '{"ok":true}'
    request = json.loads(connections[1].calls[0][2])
    assert request["format"] == schema


def test_ollama_streaming_yields_text_and_stops_on_done_event():
    listing = {"models": [{"name": "qwen2.5:7b"}]}
    stream = FakeResponse(lines=[
        b'{"message":{"content":"hello"},"done":false}\n',
        b'{"message":{"content":" world"},"done":true}\n',
    ])
    connections = [
        FakeConnection(FakeResponse(listing)),
        FakeConnection(stream),
    ]
    with patch("ayu.provider.http.client.HTTPConnection", side_effect=connections):
        chunks = list(OllamaProvider(make_settings()).stream_complete([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "hello world"
    assert connections[1].calls[0][1] == "/api/chat"
