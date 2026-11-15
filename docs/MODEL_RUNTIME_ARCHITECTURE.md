# P1 — Local Model Runtime Architecture

## Objective

Make the inference backend replaceable without coupling chat, agents, or coding
workflows to one vendor/runtime. Local mode must remain local: a failed or
misconfigured adapter must raise a visible error and must never select a remote
provider implicitly.

## Runtime contract (implemented)

The `ayu.runtime` module defines structural interfaces:

- `ChatRuntime`: the minimum `complete(messages)` contract used by chat and
  agent orchestration.
- `StreamingRuntime`: optional `stream_complete(messages)` capability. The
  terminal selects streaming by capability rather than by concrete adapter
  class.
- `StructuredOutputRuntime`: optional JSON-Schema output capability required
  by the coding workflow.
- `RuntimeDescriptor`: a safe, non-secret description of backend identity,
  local-only status, and optional capabilities. Describing a runtime does not
  contact a server or download a model.

The existing `llama-cli` and `llama-server` adapters conform to this contract.
Their transport-specific validation, timeouts, response limits, and model checks
remain inside their adapters. The coding workflow continues to reject adapters
that do not implement structured output.

## Security invariants

1. Backend selection is explicit in configuration.
2. Local adapters use local transports only; no remote fallback is permitted.
3. No adapter downloads model weights automatically.
4. Runtime errors must not include prompts, credentials, or raw provider bodies.
5. Each adapter must bound request/response size and enforce a timeout.
6. Model identity must be checked where the backend exposes enough information.
7. Optional capabilities must be detected structurally, not by checking a
   concrete provider class.
8. Runtime metadata must never expose API keys or arbitrary environment values.

## Planned adapter sequence

1. **llama.cpp CLI and llama-server** — current adapters; contract conformance
   tests are included in this milestone.
2. **Ollama** — local-only native API adapter, exact model-tag selection, no
   pull/download action, bounded requests and responses.
3. **vLLM / OpenAI-compatible local server** — local-only OpenAI-compatible
   adapter with strict loopback endpoint validation and exact model selection.
4. **AyuLM** — a native adapter only after the model's inference interface and
   serialization format are stable. Training/checkpoint commands are not
   interchangeable with chat inference.
5. **Runtime verification and evaluation** — explicit health checks, model
   identity, context limits where verifiable, throughput and task-quality
   benchmarks. Reports must distinguish measured facts from backend claims.

## Acceptance gates for each future adapter

- Unit tests for malformed messages, invalid configuration, transport failure,
  timeout, oversized responses, malformed JSON, empty output, and model mismatch.
- Tests proving that local mode never calls a remote endpoint.
- No credentials in child process environments unless the adapter explicitly
  requires a dedicated local credential mechanism.
- Structured output must be schema-constrained by the backend, not merely
  requested in the prompt.
- Streaming cancellation must close the transport and must not persist partial
  assistant output as a completed answer.
- CI on supported Python versions and operating systems.
- Live inference smoke tests are opt-in and must report the exact backend/model;
  mocked tests are not evidence of model quality or successful model loading.

## Current limitations

This milestone does not add Ollama or vLLM adapters, model downloads, automatic
model recommendation, or cross-platform live inference verification. The
existing llama-server model/context checks are stronger than the llama-cli
checks; the latter cannot prove chat-template token counts. Local inference
binaries and model parsers run with the current user's privileges and are not
OS sandboxes. Android/Termux remains unverified.
