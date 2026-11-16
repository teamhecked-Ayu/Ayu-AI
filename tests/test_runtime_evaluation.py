from pathlib import Path

import pytest

from ayu.evaluation import benchmark_chat_runtime


class FakeRuntime:
    runtime_name = "fake-local"

    def __init__(self, delays=None):
        self.settings = type("SettingsStub", (), {"local_model_id": "test-model", "local_model_path": None})()
        self.delays = list(delays or [])
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        assert messages[-1]["content"] == "Reply with exactly: AYU_RUNTIME_BENCHMARK_OK"
        if self.delays:
            import time
            time.sleep(self.delays.pop(0))
        return "AYU_RUNTIME_BENCHMARK_OK"


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
