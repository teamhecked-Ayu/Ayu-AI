# Local Model Runtime Architecture

## Objective

Make inference backends replaceable without coupling chat, agents, or coding
workflows to one provider. Local mode remains local: a failed or misconfigured
adapter raises a visible error and never selects a remote provider implicitly.

## Runtime contract

The `ayu.runtime` module defines structural `ChatRuntime`,
`StreamingRuntime`, and `StructuredOutputRuntime` interfaces plus a safe
`RuntimeDescriptor`. llama.cpp, Ollama, and vLLM adapters implement the
relevant capabilities. Adapter transports enforce their own validation,
timeouts, response limits, and model checks.

## Security invariants

1. Backend selection is explicit in configuration.
2. Local adapters use local transports only; no remote fallback is permitted.
3. No adapter downloads model weights automatically.
4. Runtime errors must not include prompts, credentials, or raw provider bodies.
5. Each adapter bounds request/response size and enforces a timeout.
6. Model identity is checked where the backend exposes enough information.
7. Optional capabilities are detected structurally.
8. Configured model identity is not described as verified loaded identity unless
   the adapter performs a successful backend identity check.

## Runtime verification and evaluation

- `ayu models verify` performs backend preflight.
- `ayu models verify --infer` performs an explicit local generation smoke test.
- `ayu models benchmark --runs 3 --warmup 1` measures end-to-end latency using
  a fixed synthetic prompt.
- `ayu models evaluate` runs
  `ayu-local-quality-expanded-v2` by default.
- `ayu models evaluate --suite ayu-local-quality-smoke-v1` retains the original
  four-case suite for historical comparison.

The original `ayu-local-quality-smoke-v1` suite remains unchanged. The
expanded `ayu-local-quality-expanded-v2` suite adds ten deterministic checks:
exact instruction following, two arithmetic items, exact formatting, strict
JSON shape/types, simple classification, exact Python function text, one
Gujarati-script response, short-context key retrieval, and a simple ordering
constraint. The report contains suite ID, per-case outcomes, category scores,
and latency. It excludes prompts, generated responses, and exception details.

Scores are comparable only within the same suite ID and documented runtime
conditions. The Gujarati item tests one fixed string, not broad multilingual
competence. The code item checks exact text, not execution or security. The
context item is short and does not measure long-context recall. Neither suite
measures factuality, safety, broad reasoning, or general intelligence. A
completed run means cases executed; it does not mean the model passed them.

## Acceptance gates for future adapters

- Unit tests for malformed messages, invalid configuration, transport failure,
  timeout, oversized responses, malformed JSON, empty output, and model mismatch.
- Tests proving local mode never calls a remote endpoint.
- No credentials in child process environments unless explicitly required.
- Structured output is schema-constrained by the backend, not merely requested
  in the prompt.
- Streaming cancellation closes the transport and never persists partial
  output as a completed answer.
- CI on supported Python versions and operating systems.
- Live inference tests are opt-in and report the exact backend/model; mocked
  tests do not prove live model loading or quality.

## Current limitations

The evaluation suites are smoke/evaluation sets, not comprehensive benchmarks.
They do not measure broad coding competence, factuality, safety, long-context
performance, or cross-platform live inference. vLLM discovery verifies served
model IDs via `/v1/models`; it does not establish weight provenance, GPU
health, or model quality. Local inference binaries and model parsers run with
the current user's privileges and are not OS sandboxes. Android/Termux remains
unverified.
