import pytest

from ayu.evaluation import (
    QUALITY_SUITE_ID,
    QUALITY_SUITE_V1_ID,
    QUALITY_SUITE_V2_ID,
    benchmark_chat_runtime,
    evaluate_chat_runtime,
)
from ayu.cli import build_parser


class FakeRuntime:
    runtime_name = "fake-local"

    def __init__(self, delays=None):
        self.settings = type("SettingsStub", (), {"local_model_id": "test-model", "local_model_path": None})()
        self.delays = list(delays or [])
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "AYU_RUNTIME_BENCHMARK_OK" in prompt:
            answer = "AYU_RUNTIME_BENCHMARK_OK"
        elif "AYU_EVAL_EXACT_OK" in prompt:
            answer = "AYU_EVAL_EXACT_OK"
        elif "23 multiplied by 19" in prompt:
            answer = "437"
        elif "exactly three lines" in prompt:
            answer = "alpha\nbeta\ngamma"
        elif "valid JSON object" in prompt:
            answer = '{"status":"ok","count":3}'
        elif "144 divided by 12" in prompt:
            answer = "12"
        elif "Classify this message" in prompt:
            answer = "spam"
        elif "Write exactly this Python function" in prompt:
            answer = "def add(a, b):\n    return a + b"
        elif "Gujarati-script greeting" in prompt:
            answer = "નમસ્તે"
        elif "PROJECT_KEY" in prompt:
            answer = "AYU-7319"
        elif "Which event happens last" in prompt:
            answer = "C"
        else:
            answer = "unexpected"
        if self.delays:
            import time
            time.sleep(self.delays.pop(0))
        return answer


def test_benchmark_reports_latency_without_output_content():
    runtime = FakeRuntime()
    result = benchmark_chat_runtime(runtime, runs=2, warmup=1)
    assert result["status"] == "passed"
    assert result["backend"] == "fake-local"
    assert result["configured_model_identity"] == "test-model"
    assert result["measured_runs"] == 2
    assert len(result["latency_ms"]["samples"]) == 2
    assert result["latency_ms"]["minimum"] <= result["latency_ms"]["median"] <= result["latency_ms"]["maximum"]
    assert runtime.calls == 3
    assert "AYU_RUNTIME_BENCHMARK_OK" not in str(result)


@pytest.mark.parametrize("runs", [0, 11, True, 1.5])
def test_benchmark_rejects_invalid_run_count(runs):
    with pytest.raises(ValueError, match="runs"):
        benchmark_chat_runtime(FakeRuntime(), runs=runs)


@pytest.mark.parametrize("warmup", [-1, 3, True, 1.5])
def test_benchmark_rejects_invalid_warmup_count(warmup):
    with pytest.raises(ValueError, match="warmup"):
        benchmark_chat_runtime(FakeRuntime(), warmup=warmup)


def test_benchmark_rejects_empty_runtime_response():
    class EmptyRuntime(FakeRuntime):
        def complete(self, messages):
            return " "
    with pytest.raises(RuntimeError, match="empty"):
        benchmark_chat_runtime(EmptyRuntime(), warmup=0)


def test_quality_v1_remains_unchanged_and_privacy_safe():
    runtime = FakeRuntime()
    report = evaluate_chat_runtime(runtime)
    assert report["status"] == "completed"
    assert report["suite_id"] == QUALITY_SUITE_V1_ID == QUALITY_SUITE_ID
    assert report["score"] == {"passed": 4, "total": 4, "percent": 100.0}
    assert {item["outcome"] for item in report["cases"]} == {"passed"}
    assert runtime.calls == 4
    assert "AYU_EVAL_EXACT_OK" not in str(report)
    assert "23 multiplied by 19" not in str(report)
    assert "status" not in report["cases"][0]


def test_expanded_v2_covers_ten_cases_and_category_scores():
    runtime = FakeRuntime()
    report = evaluate_chat_runtime(runtime, suite_id=QUALITY_SUITE_V2_ID)
    assert report["suite_id"] == "ayu-local-quality-expanded-v2"
    assert report["score"] == {"passed": 10, "total": 10, "percent": 100.0}
    assert runtime.calls == 10
    assert "multilingual" in report["category_scores"]
    assert "context_retrieval" in report["category_scores"]
    assert "AYU-7319" not in str(report)
    assert "PROJECT_KEY" not in str(report)


def test_quality_suite_rejects_unknown_version():
    with pytest.raises(ValueError, match="Unknown quality suite"):
        evaluate_chat_runtime(FakeRuntime(), suite_id="ayu-local-quality-v99")


def test_quality_suite_scores_wrong_answer_without_failing_execution():
    class WrongRuntime(FakeRuntime):
        def complete(self, messages):
            self.calls += 1
            return "I think the answer is probably 437."
    report = evaluate_chat_runtime(WrongRuntime())
    assert report["status"] == "completed"
    assert report["score"]["passed"] == 0
    assert all(item["outcome"] == "incorrect" for item in report["cases"])


def test_quality_suite_requires_exact_json_types_and_shape():
    class LooseJsonRuntime(FakeRuntime):
        def complete(self, messages):
            self.calls += 1
            if "valid JSON object" in messages[-1]["content"]:
                return '{"status":"ok","count":true}'
            return super().complete(messages)
    report = evaluate_chat_runtime(LooseJsonRuntime())
    json_case = next(item for item in report["cases"] if item["case_id"] == "structured.json_object")
    assert json_case["passed"] is False


def test_quality_suite_contains_runtime_errors_without_leaking_exception_text():
    class ErrorRuntime(FakeRuntime):
        def complete(self, messages):
            raise RuntimeError("sensitive prompt or server body")
    report = evaluate_chat_runtime(ErrorRuntime())
    assert report["score"]["passed"] == 0
    assert all(item["outcome"] == "runtime_error" for item in report["cases"])
    assert "sensitive prompt or server body" not in str(report)


def test_quality_suite_rejects_duplicate_json_keys():
    class DuplicateKeyRuntime(FakeRuntime):
        def complete(self, messages):
            self.calls += 1
            if "valid JSON object" in messages[-1]["content"]:
                return '{"status":"ok","count":1,"count":3}'
            return super().complete(messages)
    report = evaluate_chat_runtime(DuplicateKeyRuntime())
    json_case = next(item for item in report["cases"] if item["case_id"] == "structured.json_object")
    assert json_case["passed"] is False


def test_cli_exposes_bounded_local_runtime_benchmark():
    args = build_parser().parse_args(["models", "benchmark", "--runs", "2", "--warmup", "0"])
    assert args.command == "models"
    assert args.models_command == "benchmark"
    assert args.runs == 2
    assert args.warmup == 0


def test_cli_defaults_to_expanded_evaluation_and_allows_v1():
    args = build_parser().parse_args(["models", "evaluate"])
    assert args.models_command == "evaluate"
    assert args.suite == QUALITY_SUITE_V2_ID
    old = build_parser().parse_args(["models", "evaluate", "--suite", QUALITY_SUITE_V1_ID])
    assert old.suite == QUALITY_SUITE_V1_ID
