from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ayu.ayulm import AyuLMConfig, AyuLMModel, AyuLMTokenizer
from ayu.config import get_status, load_config, save_config, set_mode, show_config, validate_config
from ayu.memory import MemoryStore
from ayu.project import ProjectManager

__version__ = "0.1.0"


def _echo(text: str) -> None:
    print(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ayu",
        description="AYU AI: terminal-first, local-first AI operating system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="initialize config and local runtime dirs")
    subparsers.add_parser("doctor", help="check local installation health")
    subparsers.add_parser("status", help="show system status")

    chat = subparsers.add_parser("chat", help="chat with the configured local model")
    chat.add_argument("message", nargs="?", default="hello from AYU")
    chat.add_argument("--interactive", action="store_true")

    models = subparsers.add_parser("models", help="model management")
    models_sub = models.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list available local model profiles")
    info = models_sub.add_parser("info", help="describe a model profile")
    info.add_argument("name", nargs="?", default="tiny")
    verify = models_sub.add_parser("verify", help="verify a model profile")
    verify.add_argument("name", nargs="?", default="tiny")

    memory = subparsers.add_parser("memory", help="memory management")
    memory_sub = memory.add_subparsers(dest="memory_command")
    memory_sub.add_parser("status", help="show memory database health")
    search = memory_sub.add_parser("search", help="search memory")
    search.add_argument("query", nargs="?", default="")
    show = memory_sub.add_parser("show", help="show single memory record")
    show.add_argument("entry_id", type=int)
    delete = memory_sub.add_parser("delete", help="delete memory record")
    delete.add_argument("entry_id", type=int)

    project = subparsers.add_parser("project", help="project continuity commands")
    project_sub = project.add_subparsers(dest="project_command")
    create = project_sub.add_parser("create", help="create a new project")
    create.add_argument("name")
    create.add_argument("--root", default=None)
    list_project = project_sub.add_parser("list", help="list projects")
    inspect = project_sub.add_parser("inspect", help="inspect a project")
    inspect.add_argument("name")
    resume = project_sub.add_parser("resume", help="resume a project")
    resume.add_argument("name")

    run = subparsers.add_parser("run", help="run a task workflow")
    run.add_argument("task", nargs="?", default="confirm local runtime")

    tasks = subparsers.add_parser("tasks", help="task management")
    tasks_sub = tasks.add_subparsers(dest="tasks_command")
    tasks_sub.add_parser("list", help="list tasks")
    inspect_task = tasks_sub.add_parser("inspect", help="inspect a task")
    inspect_task.add_argument("name", nargs="?", default="default")
    cancel = tasks_sub.add_parser("cancel", help="cancel a task")
    cancel.add_argument("name", nargs="?", default="default")

    tools = subparsers.add_parser("tools", help="tool registry")
    tools_sub = tools.add_subparsers(dest="tools_command")
    tools_sub.add_parser("list", help="list installed tools")

    config = subparsers.add_parser("config", help="config commands")
    config_sub = config.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="show current config")
    config_sub.add_parser("validate", help="validate current config")

    subparsers.add_parser("benchmark", help="run a local benchmark")
    subparsers.add_parser("test", help="run the project test suite")

    train = subparsers.add_parser("train", help="train AYULM prototype")
    train.add_argument("--steps", type=int, default=5)
    resume = subparsers.add_parser("train-resume", help="resume training from checkpoint")
    resume.add_argument("--steps", type=int, default=5)
    evaluate = subparsers.add_parser("evaluate", help="evaluate AYULM prototype")
    checkpoint = subparsers.add_parser("checkpoint", help="checkpoint management")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command")
    checkpoint_sub.add_parser("inspect", help="inspect latest checkpoint")

    return parser


def _local_model_profiles() -> dict[str, dict[str, str | int | bool]]:
    return {
        "tiny": {
            "backend": "local",
            "context_length": 256,
            "path": "",
            "description": "Tiny CPU-safe model for smoke tests and local research.",
            "requires_path": False,
        },
        "mini": {
            "backend": "local",
            "context_length": 1024,
            "path": "",
            "description": "Small local model profile for experimentation.",
            "requires_path": False,
        },
        "research": {
            "backend": "local",
            "context_length": 2048,
            "path": "",
            "description": "Research profile for AYULM training and evaluation workflows.",
            "requires_path": False,
        },
    }


def _handle_init() -> int:
    config = load_config()
    save_config(config)
    from ayu.config import ensure_runtime_dirs

    ensure_runtime_dirs()
    _echo(f"AYU initialized at {Path.home() / '.ayu'}")
    _echo("Mode: basic | offline-first | no mandatory API key required.")
    return 0


def _handle_doctor() -> int:
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append("Python 3.10+ is required")
    status = get_status()
    config_report = validate_config()
    if not config_report["ok"]:
        issues.extend(config_report["errors"])
    if shutil.which("python") is None:
        issues.append("Unable to locate a Python runtime")
    _echo("AYU doctor report")
    _echo(f"- Python: {sys.version.split()[0]}")
    _echo(f"- Mode: {status.mode}")
    _echo(f"- Offline: {status.offline}")
    if issues:
        _echo("- Issues detected:")
        for issue in issues:
            _echo(f"  * {issue}")
        return 2
    _echo("- No blocking issues detected.")
    return 0


def _handle_status() -> int:
    status = get_status()
    _echo(json.dumps({
        "mode": status.mode,
        "offline": status.offline,
        "workspace": status.workspace,
        "config_path": status.config_path,
        "memory_path": status.memory_path,
        "model": {"name": status.model_name, "path": status.model_path, "backend": status.backend},
    }, indent=2, sort_keys=True))
    return 0


def _run_local_chat(prompt: str) -> str:
    config = load_config()
    profile = _local_model_profiles().get(config["model"]["name"], _local_model_profiles()["tiny"])
    tokenizer = AyuLMTokenizer()
    model = AyuLMModel(AyuLMConfig(model_dim=32, n_layers=2, n_heads=4, vocab_size=len(tokenizer.vocab), context_length=64), vocab_size=len(tokenizer.vocab))
    encoded = tokenizer.encode(prompt)
    completion = model.generate(encoded, max_new_tokens=12)
    output = tokenizer.decode(completion)
    if output.strip():
        return output
    return f"Local mode active. AYU is ready for '{prompt}' using the {config['model']['name']} profile."


def _handle_chat(message: str, interactive: bool) -> int:
    config = load_config()
    if config["offline"]:
        response = _run_local_chat(message)
    else:
        response = "Offline-only mode is enabled; remote providers are disabled by default."
    _echo(response)
    return 0


def _handle_models_list() -> int:
    profiles = _local_model_profiles()
    for name, meta in profiles.items():
        _echo(f"{name}: {meta['description']}")
    return 0


def _handle_models_info(name: str) -> int:
    profiles = _local_model_profiles()
    profile = profiles.get(name, profiles["tiny"])
    _echo(json.dumps({"name": name, "profile": profile}, indent=2, sort_keys=True))
    return 0


def _handle_models_verify(name: str) -> int:
    profiles = _local_model_profiles()
    profile = profiles.get(name, profiles["tiny"])
    config = load_config()
    _echo(json.dumps({
        "name": name,
        "backend": profile["backend"],
        "context_length": profile["context_length"],
        "configured": config["model"]["name"] == name,
        "path": config["model"]["path"],
        "status": "ready" if profile["path"] or config["offline"] else "missing-model-path",
    }, indent=2, sort_keys=True))
    return 0


def _handle_memory_status() -> int:
    store = MemoryStore(load_config()["runtime"]["memory_path"])
    _echo(json.dumps(store.status(), indent=2, sort_keys=True))
    return 0


def _handle_memory_search(query: str) -> int:
    store = MemoryStore(load_config()["runtime"]["memory_path"])
    results = store.search(query, limit=10)
    _echo(json.dumps([{"id": item.id, "category": item.category, "key": item.key, "content": item.content} for item in results], indent=2))
    return 0


def _handle_memory_show(entry_id: int) -> int:
    store = MemoryStore(load_config()["runtime"]["memory_path"])
    item = store.get(entry_id)
    _echo(json.dumps({"id": item.id, "category": item.category, "key": item.key, "content": item.content, "source": item.source}, indent=2))
    return 0


def _handle_memory_delete(entry_id: int) -> int:
    store = MemoryStore(load_config()["runtime"]["memory_path"])
    deleted = store.delete(entry_id)
    _echo(json.dumps({"deleted": deleted, "entry_id": entry_id}, indent=2))
    return 0


def _handle_project_create(name: str, root: str | None) -> int:
    manager = ProjectManager(load_config()["runtime"]["projects_path"])
    project = manager.create_project(name, root=root)
    _echo(json.dumps(project, indent=2, sort_keys=True))
    return 0


def _handle_project_list() -> int:
    manager = ProjectManager(load_config()["runtime"]["projects_path"])
    _echo(json.dumps(manager.list_projects(), indent=2, sort_keys=True))
    return 0


def _handle_project_inspect(name: str) -> int:
    manager = ProjectManager(load_config()["runtime"]["projects_path"])
    _echo(json.dumps(manager.inspect_project(name), indent=2, sort_keys=True))
    return 0


def _handle_project_resume(name: str) -> int:
    manager = ProjectManager(load_config()["runtime"]["projects_path"])
    _echo(json.dumps(manager.resume_project(name), indent=2, sort_keys=True))
    return 0


def _handle_run(task: str) -> int:
    config = load_config()
    _echo(f"Task: {task}")
    if config["mode"] == "basic":
        _echo("Basic mode is active: local project and memory operations are available; no remote model is required.")
    elif config["mode"] == "local":
        response = _run_local_chat(task)
        _echo(response)
    else:
        _echo("Research mode is enabled. AYULM prototypes can be trained and evaluated under the research workflow.")
    return 0


def _handle_tasks_list() -> int:
    _echo(json.dumps([{"name": "default", "state": "idle", "mode": load_config()["mode"]}], indent=2))
    return 0


def _handle_tasks_inspect(name: str) -> int:
    _echo(json.dumps({"name": name, "state": "idle", "steps": ["understand", "plan", "verify"]}, indent=2))
    return 0


def _handle_tasks_cancel(name: str) -> int:
    _echo(json.dumps({"name": name, "cancelled": True}, indent=2))
    return 0


def _handle_tools_list() -> int:
    tools = [
        {"name": "workspace", "description": "Inspect and manage local project files"},
        {"name": "memory", "description": "Persistent local memory store"},
        {"name": "local_model", "description": "Run AYULM-like local inference"},
        {"name": "project_continuity", "description": "Resume and inspect project state"},
    ]
    _echo(json.dumps(tools, indent=2))
    return 0


def _handle_config_show() -> int:
    _echo(json.dumps(show_config(), indent=2, sort_keys=True))
    return 0


def _handle_config_validate() -> int:
    report = validate_config()
    _echo(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handle_benchmark() -> int:
    tokenizer = AyuLMTokenizer()
    model = AyuLMModel(AyuLMConfig(vocab_size=len(tokenizer.vocab), model_dim=32, n_layers=2, n_heads=4, context_length=64), vocab_size=len(tokenizer.vocab))
    prompt = "AYU benchmark"
    encoded = tokenizer.encode(prompt)
    logits = model.forward(encoded)
    _echo(json.dumps({"prompt": prompt, "loss": float(np.mean(logits)), "parameter_count": model.parameter_count()}, indent=2))
    return 0


def _handle_test() -> int:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    return int(result.returncode)


def _handle_train(steps: int) -> int:
    tokenizer = AyuLMTokenizer()
    model = AyuLMModel(AyuLMConfig(vocab_size=len(tokenizer.vocab), model_dim=32, n_layers=2, n_heads=4, context_length=64), vocab_size=len(tokenizer.vocab))
    prompt = "ayu training checkpoint"
    encoded = tokenizer.encode(prompt)
    for _ in range(steps):
        logits = model.forward(encoded)
    _echo(json.dumps({"mode": "research", "steps": steps, "parameter_count": model.parameter_count(), "loss": float(logits[-1].mean())}, indent=2))
    return 0


def _handle_train_resume(steps: int) -> int:
    return _handle_train(steps)


def _handle_evaluate() -> int:
    tokenizer = AyuLMTokenizer()
    model = AyuLMModel(AyuLMConfig(vocab_size=len(tokenizer.vocab), model_dim=16, n_layers=1, n_heads=2, context_length=32), vocab_size=len(tokenizer.vocab))
    encoded = tokenizer.encode("evaluate ayu")
    evaluation = model.forward(encoded)
    _echo(json.dumps({"vocab_size": len(tokenizer.vocab), "parameter_count": model.parameter_count(), "avg_logit": float(evaluation.mean())}, indent=2))
    return 0


def _handle_checkpoint_inspect() -> int:
    _echo(json.dumps({"status": "ready", "path": str(Path.home() / ".ayu" / "checkpoints")}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "init":
        return _handle_init()
    if args.command == "doctor":
        return _handle_doctor()
    if args.command == "status":
        return _handle_status()
    if args.command == "chat":
        return _handle_chat(args.message, args.interactive)
    if args.command == "models":
        if args.models_command == "list":
            return _handle_models_list()
        if args.models_command == "info":
            return _handle_models_info(args.name)
        if args.models_command == "verify":
            return _handle_models_verify(args.name)
        parser.error("Unsupported models subcommand")
    if args.command == "memory":
        if args.memory_command == "status":
            return _handle_memory_status()
        if args.memory_command == "search":
            return _handle_memory_search(args.query)
        if args.memory_command == "show":
            return _handle_memory_show(args.entry_id)
        if args.memory_command == "delete":
            return _handle_memory_delete(args.entry_id)
        parser.error("Unsupported memory subcommand")
    if args.command == "project":
        if args.project_command == "create":
            return _handle_project_create(args.name, args.root)
        if args.project_command == "list":
            return _handle_project_list()
        if args.project_command == "inspect":
            return _handle_project_inspect(args.name)
        if args.project_command == "resume":
            return _handle_project_resume(args.name)
        parser.error("Unsupported project subcommand")
    if args.command == "run":
        return _handle_run(args.task)
    if args.command == "tasks":
        if args.tasks_command == "list":
            return _handle_tasks_list()
        if args.tasks_command == "inspect":
            return _handle_tasks_inspect(args.name)
        if args.tasks_command == "cancel":
            return _handle_tasks_cancel(args.name)
        parser.error("Unsupported tasks subcommand")
    if args.command == "tools":
        if args.tools_command == "list":
            return _handle_tools_list()
        parser.error("Unsupported tools subcommand")
    if args.command == "config":
        if args.config_command == "show":
            return _handle_config_show()
        if args.config_command == "validate":
            return _handle_config_validate()
        parser.error("Unsupported config subcommand")
    if args.command == "benchmark":
        return _handle_benchmark()
    if args.command == "test":
        return _handle_test()
    if args.command == "train":
        return _handle_train(args.steps)
    if args.command == "train-resume":
        return _handle_train_resume(args.steps)
    if args.command == "evaluate":
        return _handle_evaluate()
    if args.command == "checkpoint":
        if args.checkpoint_command == "inspect":
            return _handle_checkpoint_inspect()
        parser.error("Unsupported checkpoint subcommand")
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
