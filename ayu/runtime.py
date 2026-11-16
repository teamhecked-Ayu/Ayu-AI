"""Provider-neutral contracts for local model runtimes.

Adapters remain responsible for transport-specific validation and timeouts. This
module defines the stable surface consumed by chat, agents, and coding flows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


Message = dict[str, str]


@runtime_checkable
class ChatRuntime(Protocol):
    """Minimum interface required by ordinary Ayu chat and agent workflows."""

    runtime_name: str

    def complete(self, messages: list[Message]) -> str:
        """Return one non-empty assistant response or raise ProviderError."""


@runtime_checkable
class StructuredOutputRuntime(Protocol):
    """Optional capability required by schema-constrained coding workflows."""

    def complete_structured(
        self, messages: list[Message], schema: dict, *, schema_name: str
    ) -> str:
        """Return a response conforming to the supplied JSON Schema."""


@runtime_checkable
class StreamingRuntime(Protocol):
    """Optional capability for incremental terminal output."""

    def stream_complete(self, messages: list[Message]):
        """Yield text chunks; closing the iterator must close the transport."""


@dataclass(frozen=True)
class RuntimeDescriptor:
    """Safe, non-secret description of the selected local runtime."""

    backend: str
    locality: str
    supports_streaming: bool
    supports_structured_output: bool
    configured_model_identity: str | None = None


def describe_runtime(runtime: ChatRuntime) -> RuntimeDescriptor:
    """Describe capabilities without probing, downloading, or contacting a server."""
    name = getattr(runtime, "runtime_name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Local runtime adapter must expose a non-empty runtime_name.")
    settings = getattr(runtime, "settings", None)
    model_id = getattr(settings, "local_model_id", None)
    model_path = getattr(settings, "local_model_path", None)
    identity = model_id or (str(model_path) if model_path is not None else None)
    return RuntimeDescriptor(
        backend=name,
        locality="local",
        supports_streaming=callable(getattr(runtime, "stream_complete", None)),
        supports_structured_output=callable(getattr(runtime, "complete_structured", None)),
        configured_model_identity=identity,
    )
