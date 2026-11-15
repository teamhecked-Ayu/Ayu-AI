from pathlib import Path

from ayu.provider import LlamaCppProvider, LlamaServerProvider
from ayu.runtime import ChatRuntime, StructuredOutputRuntime, StreamingRuntime, describe_runtime
from ayu.settings import Settings


def settings(**overrides):
    values = {
        "api_key": None,
        "base_url": "https://api.example.test/v1",
        "model": None,
        "database_path": Path("/tmp/ayu-runtime-test.sqlite"),
        "local_model_path": Path("/models/example.gguf"),
    }
    values.update(overrides)
    return Settings(**values)


def test_llama_cli_conforms_to_minimum_chat_runtime_contract():
    provider = LlamaCppProvider(settings())
    assert isinstance(provider, ChatRuntime)
    descriptor = describe_runtime(provider)
    assert descriptor.backend == "llama-cli"
    assert descriptor.locality == "local"
    assert descriptor.configured_model_identity.endswith("example.gguf")
    assert descriptor.supports_streaming is False
    assert descriptor.supports_structured_output is False


def test_llama_server_exposes_optional_runtime_capabilities():
    provider = LlamaServerProvider(settings(local_model_id="qwen-local"))
    assert isinstance(provider, ChatRuntime)
    assert isinstance(provider, StructuredOutputRuntime)
    assert isinstance(provider, StreamingRuntime)
    descriptor = describe_runtime(provider)
    assert descriptor.backend == "llama-server"
    assert descriptor.locality == "local"
    assert descriptor.configured_model_identity == "qwen-local"
    assert descriptor.supports_streaming is True
    assert descriptor.supports_structured_output is True


def test_runtime_description_does_not_require_server_probe_or_model_download():
    provider = LlamaServerProvider(settings())
    descriptor = describe_runtime(provider)
    assert descriptor.backend == "llama-server"
    assert descriptor.configured_model_identity.endswith("example.gguf")
