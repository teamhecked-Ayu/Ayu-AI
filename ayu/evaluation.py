"""Privacy-safe local runtime latency and task-quality evaluations.

The quality suite is a small deterministic smoke suite, not a broad measure of
reasoning, factuality, coding ability, or general intelligence.
"""
from __future__ import annotations

import json
import statistics
import time
from typing import Any, Callable

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


def _exact_text(expected: str) -> Callable[[str], bool]:
    return lambda answer: answer.strip() == expected


def _exact_json(expected: dict[str, Any]) -> Callable[[str], bool]:
    def validate(answer: str) -> bool:
        try:
            parsed = json.loads(answer)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(parsed, dict) or parsed.keys() != expected.keys():
            return False
        return all(type(parsed[key]) is type(value) and parsed[key] == value
                   for key, value in expected.items())
    return validate


QUALITY_SUITE_ID = "ayu-local-quality-smoke-v1"
_QUALITY_CASES: tuple[tuple[str, str, str, Callable[[str], bool]], ...] = (
    (
        "instruction.exact_token",
        "instruction_following",
        "Reply with exactly this token and nothing else: AYU_EVAL_EXACT_OK",
        _exact_text("AYU_EVAL_EXACT_OK"),
    ),
    (
        "reasoning.integer_arithmetic",
        "basic_arithmetic",
        "Calculate 23 multiplied by 19. Reply with only the integer.",
        _exact_text("437"),
    ),
    (
        "format.exact_lines",
        "format_adherence",
        "Return exactly three lines, with no bullets or extra text:\nalpha\nbeta\ngamma",
        _exact_text("alpha\nbeta\ngamma"),
    ),
    (
        "structured.json_object",
        "structured_output",
        'Return only a valid JSON object with exactly two keys: "status" set to the string "ok", and "count" set to the integer 3.',
        _exact_json({"status": "ok", "count": 3}),
    ),
)


def evaluate_chat_runtime(runtime: ChatRuntime) -> dict[str, Any]:
    """Run a fixed synthetic suite; omit prompts, responses, and exception details."""
    results: list[dict[str, Any]] = []
    category_totals: dict[str, list[int]] = {}
    elapsed: list[float] = []
    system_message = (
        "You are completing a deterministic local model evaluation. "
        "Follow the user's output constraints exactly. Do not add explanations unless requested."
    )

    for case_id, category, prompt, validator in _QUALITY_CASES:
        started = time.perf_counter()
        try:
            answer = runtime.complete([
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ])
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            if not isinstance(answer, str):
                passed = False
                outcome = "invalid_response_type"
            elif not answer.strip():
                passed = False
                outcome = "empty_response"
            else:
                passed = bool(validator(answer))
                outcome = "passed" if passed else "incorrect"
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            passed = False
            outcome = "runtime_error"

        elapsed.append(duration_ms)
        category_totals.setdefault(category, []).append(1 if passed else 0)
        results.append({
            "case_id": case_id,
            "category": category,
            "passed": passed,
            "outcome": outcome,
            "latency_ms": duration_ms,
        })

    passed_count = sum(item["passed"] for item in results)
    total = len(results)
    descriptor = describe_runtime(runtime)
    return {
        "status": "completed",
        "suite_id": QUALITY_SUITE_ID,
        "backend": descriptor.backend,
        "locality": descriptor.locality,
        "configured_model_identity": descriptor.configured_model_identity,
        "model_identity_note": "Configured identity only; this report does not independently verify loaded weights.",
        "score": {
            "passed": passed_count,
            "total": total,
            "percent": round((passed_count / total) * 100, 1),
        },
        "category_scores": {
            category: {
                "passed": sum(values),
                "total": len(values),
                "percent": round((sum(values) / len(values)) * 100, 1),
            }
            for category, values in sorted(category_totals.items())
        },
        "cases": results,
        "latency_ms": {
            "total": round(sum(elapsed), 3),
            "median_per_case": round(statistics.median(elapsed), 3),
        },
        "privacy": "synthetic prompts only; prompts, generated responses, and exception details are not included",
        "limitations": [
            "This four-case smoke suite is not a comprehensive model-quality benchmark.",
            "It tests exact instruction following, one arithmetic item, exact formatting, and strict JSON shape only.",
            "It does not measure factuality, broad reasoning, coding competence, multilingual quality, safety, or long-context performance.",
            "Scores are meaningful only when comparing the same suite version under documented conditions.",
            "A completed evaluation may have a low score; completion means the cases ran, not that the model passed them.",
        ],
    }
