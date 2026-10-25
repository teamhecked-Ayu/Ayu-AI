"""Small, bounded offline self-tests; all generated data stays in a temporary directory."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .provider import remote_endpoint_error
from .store import Store


def run_smoke_test(include_ayulm: bool = False) -> int:
    """Exercise core local features; optionally run one tiny CPU AyuLM update.

    Returns 0 only when every requested check completed successfully. The
    optional model check never reads user corpora or writes persistent weights.
    """
    report: dict[str, Any] = {"mode": "offline", "checks": [], "ayulm": "not_requested"}
    try:
        with tempfile.TemporaryDirectory(prefix="ayu-selftest-") as temporary:
            root = Path(temporary)
            store = Store(root / "smoke.sqlite3")
            project_id = store.create_project("smoke-test", str(root))
            task_id = store.add_task(project_id, "verify local persistence")
            store.add_message("smoke", "user", "temporary test message")
            memory_id = store.remember("temporary searchable Ayu memory", category="semantic")
            if store.history("smoke") != [{"role": "user", "content": "temporary test message"}]:
                raise RuntimeError("chat persistence check failed")
            if not store.recall("searchable", category="semantic"):
                raise RuntimeError("lexical memory retrieval check failed")
            if not store.update_task_status(task_id, "completed") or store.list_tasks(project_id)[0]["status"] != "completed":
                raise RuntimeError("project task persistence check failed")
            if not store.forget(memory_id) or store.recall("temporary searchable"):
                raise RuntimeError("memory deletion check failed")
            report["checks"].extend(["sqlite_chat", "memory_search_delete", "project_task_persistence"])

            if remote_endpoint_error("http://example.invalid/v1") is None:
                raise RuntimeError("plaintext remote endpoint was not rejected")
            report["checks"].append("https_endpoint_policy")

            if include_ayulm:
                from . import ayulm
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError("AyuLM smoke test requested but optional PyTorch is not installed; install with `python -m pip install -e '.[training]'`.") from exc
                previous_threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(1)
                    corpus = root / "synthetic-corpus.txt"
                    corpus.write_text("AyuLM bounded CPU smoke-test corpus. " * 24, encoding="utf-8")
                    checkpoint = root / "smoke-checkpoint.pt"
                    config = ayulm.ModelConfig(seq_len=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
                    metrics = ayulm.train(
                        corpus,
                        checkpoint,
                        steps=1,
                        config=config,
                        seed=17,
                        batch_size=2,
                        eval_interval=1,
                        device="cpu",
                        registry=root / "registry.json",
                    )
                    model, metadata = ayulm.load_checkpoint(checkpoint, "cpu")
                    prompt = torch.tensor([[1, *ayulm.ByteTokenizer().encode("Ayu")]], dtype=torch.long)
                    generated = model.generate(prompt, max_new_tokens=2)
                    expected = ("train_loss", "validation_loss", "validation_perplexity")
                    if not checkpoint.is_file() or int(metadata.get("step", 0)) != 1:
                        raise RuntimeError("AyuLM checkpoint verification failed")
                    if generated.shape[1] != prompt.shape[1] + 2:
                        raise RuntimeError("AyuLM generation length check failed")
                    if any(name not in metrics or metrics[name] is None or not math.isfinite(float(metrics[name])) for name in expected):
                        raise RuntimeError("AyuLM training metrics were missing or non-finite")
                    report["ayulm"] = {
                        "status": "passed",
                        "steps": metrics["step"],
                        "parameters": metrics["parameters"],
                        "train_loss": metrics["train_loss"],
                        "validation_loss": metrics["validation_loss"],
                        "validation_perplexity": metrics["validation_perplexity"],
                        "generated_tokens": 2,
                        "checkpoint": "temporary; removed on exit",
                    }
                    report["checks"].extend(["ayulm_train", "checkpoint_load", "autoregressive_generation"])
                finally:
                    torch.set_num_threads(previous_threads)
        report["status"] = "passed"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
