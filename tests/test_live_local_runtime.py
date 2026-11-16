"""Opt-in live tests for an explicitly configured local inference server.

These tests are skipped in normal offline CI. They make real generation requests
only when AYU_RUN_LIVE_RUNTIME_TESTS=1 is explicitly set by the operator.
"""
import json
import os

import pytest

from ayu.provider import build_local_provider
from ayu.runtime import ChatRuntime, StreamingRuntime, StructuredOutputRuntime, describe_runtime
from ayu.settings import Settings


pytestmark = pytest.mark.skipif(
    os.getenv("AYU_RUN_LIVE_RUNTIME_TESTS") != "1",
    reason="set AYU_RUN_LIVE_RUNTIME_TESTS=1 to explicitly enable live local-model tests",
)


def test_selected_local_server_chat_streaming_and_structured_output():
    settings = Settings.load()
    assert settings.mode == "local", "live runtime tests require AYU_MODE=local"
    assert settings.local_backend in {"llama-server", "ollama", "vllm"}, (
        "live runtime tests require a server backend: llama-server, ollama, or vllm"
    )
    assert settings.local_model_id, "set AYU_LOCAL_MODEL_ID to the exact served model ID"

    provider = build_local_provider(settings)
    assert isinstance(provider, ChatRuntime)
    assert isinstance(provider, StreamingRuntime)
    assert isinstance(provider, StructuredOutputRuntime)

    selected_model = provider.validate()
    descriptor = describe_runtime(provider)
    assert descriptor.locality == "local"
    assert descriptor.backend == settings.local_backend
    assert descriptor.configured_model_identity == settings.local_model_id
    assert selected_model == settings.local_model_id

    chat = provider.complete([
        {"role": "system", "content": "You are running a local inference connectivity test."},
        {"role": "user", "content": "Reply with a short confirmation that local chat generation works."},
    ])
    assert isinstance(chat, str) and chat.strip()

    chunks = list(provider.stream_complete([
        {"role": "system", "content": "You are running a local streaming connectivity test."},
        {"role": "user", "content": "Reply with a short confirmation that local streaming works."},
    ]))
    assert chunks and all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunks).strip()

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
            "count": {"type": "integer"},
        },
        "required": ["status", "count"],
        "additionalProperties": False,
    }
    structured = provider.complete_structured(
        [
            {"role": "system", "content": "Return only JSON matching the supplied schema."},
            {"role": "user", "content": "Set status to ok and count to 7."},
        ],
        schema,
        schema_name="ayu_live_check",
    )
    payload = json.loads(structured)
    assert payload == {"status": "ok", "count": 7}
