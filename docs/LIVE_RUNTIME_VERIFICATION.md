# Live Local Runtime Verification

## Purpose

The regular test suite uses deterministic mocks and remains offline. This
opt-in test makes real inference requests to the already-running local server
selected by Ayu. It does not install runtimes, download models, start servers,
or contact remote providers.

The test verifies three separate capabilities against one explicitly selected
model:

1. Model discovery and exact model-ID selection.
2. Non-streaming chat generation.
3. Incremental streaming and JSON-Schema-constrained generation.

A successful run proves that these requests worked at that time on that
machine. It does not prove model quality, weight provenance, GPU health,
security isolation, or compatibility with every model.

## Configure one local server

Install and start one server yourself. Load a model that you are entitled to
use. Bind the server to loopback only. Ayu's server adapters reject remote
addresses and hostnames.

For Ollama:

```sh
export AYU_MODE=local
export AYU_LOCAL_BACKEND=ollama
export AYU_OLLAMA_URL=http://127.0.0.1:11434
export AYU_LOCAL_MODEL_ID='exact-name-from-ollama-list'
```

For vLLM:

```sh
export AYU_MODE=local
export AYU_LOCAL_BACKEND=vllm
export AYU_VLLM_URL=http://127.0.0.1:8000
export AYU_LOCAL_MODEL_ID='exact-id-returned-by-v1-models'
```

For llama-server:

```sh
export AYU_MODE=local
export AYU_LOCAL_BACKEND=llama-server
export AYU_LLAMA_SERVER_URL=http://127.0.0.1:9931
export AYU_LOCAL_MODEL='/absolute/path/to/the/already-loaded-model.gguf'
export AYU_LOCAL_MODEL_ID='exact-id-returned-by-the-server'
```

Set a reasonable timeout for the hardware and model:

```sh
export AYU_LOCAL_TIMEOUT=300
export AYU_MAX_OUTPUT_TOKENS=128
```

Then install the project and run the preflight first:

```sh
python -m pip install -e '.[dev]'
ayu models verify
```

Preflight checks connectivity and model identity; it does not generate text.
Run the live test explicitly:

```sh
AYU_RUN_LIVE_RUNTIME_TESTS=1 python -m pytest -q -s tests/test_live_local_runtime.py
```

PowerShell:

```powershell
$env:AYU_RUN_LIVE_RUNTIME_TESTS = "1"
python -m pytest -q -s tests/test_live_local_runtime.py
```

The test fails closed if Ayu is not in local mode, a server backend is not
selected, an exact model ID is not configured, model discovery fails, or any
of the three inference capabilities fails. It never switches providers.

## Follow-up checks

After the live connectivity test, run the versioned quality suite and latency
benchmark against the same configuration:

```sh
ayu models evaluate --suite ayu-local-quality-expanded-v2
ayu models benchmark --runs 3 --warmup 1
```

Keep the backend, exact model ID, model quantization, server version, hardware,
and test date with any results you compare. Evaluation reports intentionally
omit prompts and generated text. The suite is a small smoke test, not a
comprehensive model benchmark.

## CI behavior

The live test is skipped unless `AYU_RUN_LIVE_RUNTIME_TESTS=1` is explicitly
set. Standard GitHub Actions therefore remains deterministic, offline, and
does not download model weights. The test has not been executed against a
real model merely because the normal CI suite passes. Record a live result
only after running it against an installed server and model.
