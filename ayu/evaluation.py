"""Small, privacy-safe latency evaluation for an explicitly selected local chat runtime."""
from __future__ import annotations

import statistics
import time
from typing import Any

from .runtime import ChatRuntime, describe_runtime


BENCHMARK_MESSAGES = [
    {"role": "system", "content": "You are running a local runtime latency check. Answer briefly and directly."},
    {"role": "user", "content": "Reply with exactly: AYU_RUNTIME_BENCHMARK_OK"},
]


def benchmark_chat_runtime(
    runtime: ChatRuntime, *, runs: int = 3, warmup: int = 1
) -> dict[str, Any]:
    """Measure end-to-end completion latency; never records prompts or generated text."""
    if isinstance(runs, bool) or not isinstance(runs, int) or not 1 <= runs <= 10:
        raise ValueError("runs must be an integer from 1 to 10")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 2:
        raise ValueError("warmup must be an integer from 0 to 2")

    for _ in range(warmup):
        answer = runtime.complete(list(BENCHMARK_MESSAGES))
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Local runtime returned an empty or non-text warm-up response.")

    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        answer = runtime.complete(list(BENCHMARK_MESSAGES))
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Local runtime returned an empty or non-text benchmark response.")
        samples.append(round(elapsed_ms, 3))

    descriptor = describe_runtime(runtime)
    return {
        "status": "passed",
        "backend": descriptor.backend,
        "locality": descriptor.locality,
        "configured_model_identity": descriptor.configured_model_identity,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "latency_ms": {
            "samples": samples,
            "minimum": round(min(samples), 3),
            "median": round(statistics.median(samples), 3),
            "maximum": round(max(samples), 3),
        },
        "measurement": "end-to-end complete() wall time; includes adapter and local server overhead",
        "privacy": "fixed synthetic prompt; generated text and prompts are not included in this report",
        "limitations": [
            "Latency is not a measure of answer quality.",
            "Token throughput is not reported because adapters do not expose comparable token usage.",
            "Results depend on hardware, model state, server load, quantization, and warm-up settings.",
        ],
    }
