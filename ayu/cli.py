"""Terminal-first Ayu AI CLI. Basic commands work without any AI credentials."""
from __future__ import annotations
import argparse, contextlib, json, os, re, shutil, sqlite3, sys, tempfile, uuid
from pathlib import Path
from . import __version__
from .settings import Settings
from .provider import LlamaCppProvider, LlamaServerProvider, NewsAPIClient, ProviderError, build_local_provider, build_remote_provider, remote_endpoint_error
from .store import MEMORY_CATEGORIES, Store

SYSTEM = ("You are Ayu AI, a careful assistant. Be candid about uncertainty. Do not claim to have executed code, "
          "browsed the web, or changed files unless a tool actually did so. Never expose secrets. Prefer safe, reversible advice.")
ROLES = {
    "general": SYSTEM,
    "coder": SYSTEM + " Focus on correct, maintainable code. State assumptions and provide tests when useful.",
    "researcher": SYSTEM + " Distinguish verified facts from hypotheses. External research is unavailable unless explicitly configured.",
    "planner": SYSTEM + " Provide realistic steps, dependencies, risks, and verification checks.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ayu", description="Ayu AI — local-first terminal assistant; no API key needed for basic commands")
    parser.add_argument("--version", action="version", version=f"ayu {__version__}")
    sub = parser.add_subparsers(dest="command")
    init = sub.add_parser("init", help="Initialize local configuration without creating credentials")
    init.add_argument("--force", action="store_true", help="Replace existing .env configuration")
    sub.add_parser("doctor", help="Check local setup without making network requests")
    sub.add_parser("status", help="Show privacy-safe status and local database counts")
    chat = sub.add_parser("chat", help="Chat using an explicitly selected local or remote model")
    chat.add_argument("--mode", choices=("local", "remote", "basic"), help="Inference mode (default AYU_MODE or local); remote is never selected automatically")
    chat.add_argument("--session", help="Resume an existing conversation session")
    chat.add_argument("--role", choices=ROLES, default="general", help="Prompt profile")
    chat.add_argument("--once", metavar="PROMPT", help="Send one prompt and exit")
    models = sub.add_parser("models", help="Inspect models and local inference runtime")
    ms = models.add_subparsers(dest="models_command", required=True)
    ms.add_parser("list", help="List configured local/remote models")
    ms.add_parser("info", help="Show model configuration without secrets")
    model_verify = ms.add_parser("verify", help="Preflight the configured local model/runtime; use --infer to test actual generation")
    model_verify.add_argument("--infer", action="store_true", help="Load/use the selected local model and require a non-empty generated response")
    memory = sub.add_parser("memory", help="Manage private, local SQLite memories")
    mem = memory.add_subparsers(dest="memory_command", required=True)
    mem.add_parser("status", help="Show memory count and storage location")
    search = mem.add_parser("search", help="Search saved memories")
    search.add_argument("query", nargs="*", help="Optional lexical search")
    search.add_argument("--category", choices=sorted(MEMORY_CATEGORIES))
    search.add_argument("--project")
    add = mem.add_parser("add", help="Save a user-provided memory")
    add.add_argument("text", nargs="+"); add.add_argument("--category", choices=sorted(MEMORY_CATEGORIES), default="semantic"); add.add_argument("--project")
    show = mem.add_parser("show", help="Inspect one memory")
    show.add_argument("id", type=int)
    update = mem.add_parser("update", help="Update a memory")
    update.add_argument("id", type=int); update.add_argument("--text"); update.add_argument("--category", choices=sorted(MEMORY_CATEGORIES))
    delete = mem.add_parser("delete", help="Delete a memory")
    delete.add_argument("id", type=int)
    exp=mem.add_parser("export",help="Export saved memories to a local JSON file")
    exp.add_argument("output",type=Path); exp.add_argument("--force",action="store_true",help="Overwrite an existing export file")
    imp=mem.add_parser("import",help="Import and validate a local JSON memory export")
    imp.add_argument("input",type=Path)
    projects = sub.add_parser("project", help="Track projects and resume saved task state")
    ps = projects.add_subparsers(dest="project_command", required=True)
    create = ps.add_parser("create", help="Create a local project record")
    create.add_argument("name"); create.add_argument("--root", help="Optional project directory")
    ps.add_parser("list", help="List project records")
    inspect = ps.add_parser("inspect", help="Inspect saved project and task state")
    inspect.add_argument("project")
    update_project = ps.add_parser("update", help="Update a saved continuity field with a JSON value")
    update_project.add_argument("project"); update_project.add_argument("--field", required=True, choices=("requirements","goals","architecture_decisions","current_state","completed","pending","known_problems","checkpoints","artifact_references")); update_project.add_argument("--value", required=True, help="JSON value, e.g. '[\"Use SQLite\"]' or '\"In progress\"'")
    resume = ps.add_parser("resume", help="Verify saved references and show next pending project task")
    resume.add_argument("project")
    tasks = sub.add_parser("tasks", help="Manage local project task records")
    ts = tasks.add_subparsers(dest="tasks_command", required=True)
    ta = ts.add_parser("add", help="Add a task to a project")
    ta.add_argument("project"); ta.add_argument("title"); ta.add_argument("--depends-on", nargs="+", type=int, default=[], help="Existing prerequisite task ID(s) in the same project")
    tl = ts.add_parser("list", help="List project tasks")
    tl.add_argument("project")
    ti = ts.add_parser("inspect", help="Inspect a task and its dependency readiness")
    ti.add_argument("id", type=int)
    td = ts.add_parser("set-dependencies", help="Replace dependencies on a pending task; cycles and cross-project edges are rejected")
    td.add_argument("id", type=int); td.add_argument("--depends-on", nargs="*", type=int, default=[])
    for task_action, help_text in (("start", "Mark a ready task in progress"), ("complete", "Complete a task only after its prerequisites are complete")):
        action_parser=ts.add_parser(task_action,help=help_text); action_parser.add_argument("id",type=int)
    tc = ts.add_parser("cancel", help="Cancel a pending task")
    tc.add_argument("id", type=int)
    config = sub.add_parser("config", help="Inspect and validate local configuration")
    cs = config.add_subparsers(dest="config_command", required=True)
    cs.add_parser("show", help="Show configuration (credential value is redacted)")
    cs.add_parser("validate", help="Validate configuration without contacting providers")
    tools = sub.add_parser("tools", help="Show implemented tool capabilities")
    tools.add_subparsers(dest="tools_command", required=True).add_parser("list", help="List available local tools")
    agents = sub.add_parser("agents", help="Run bounded model-mediated agent workflows and inspect their audit records")
    ag = agents.add_subparsers(dest="agents_command", required=True)
    ag.add_parser("roles", help="List the fixed sequential agent roles")
    agent_run = ag.add_parser("run", help="Plan and analyze a workspace task using read-only tools")
    agent_run.add_argument("task", nargs="+", help="Task description; not executed as a shell command")
    agent_run.add_argument("--workspace", type=Path, required=True, help="Explicit existing root directory that read-only tools may access")
    agent_run.add_argument("--project", help="Optional saved project ID/name; workspace must be inside its configured root")
    agent_run.add_argument("--mode", choices=("local", "remote"), help="Selected inference mode (default AYU_MODE; no fallback)")
    agent_run.add_argument("--max-steps", type=int, default=3, help="Maximum model-proposed read-only tools (1..4)")
    agent_run.add_argument("--allow-remote-context", action="store_true", help="Explicitly allow selected local excerpts to be sent to the configured remote provider")
    agent_inspect = ag.add_parser("inspect", help="Inspect a persisted agent run and its local audit events")
    agent_inspect.add_argument("run_id")
    code = sub.add_parser("code", help="Run a bounded local-model coding task with sandboxed verification")
    code_sub = code.add_subparsers(dest="code_command", required=True)
    code_run = code_sub.add_parser("run", help="Plan, edit, test in a required local Docker sandbox, review, and preserve a local diff")
    code_run.add_argument("task", nargs="+", help="Requested code outcome; never interpreted as a shell command")
    code_run.add_argument("--workspace", type=Path, required=True, help="Clean Git repository root; changes go only to a separate temporary worktree")
    code_run.add_argument("--test-suite", choices=("pytest", "unittest", "compileall"), required=True, help="Fixed, allow-listed Python verification command")
    code_run.add_argument("--test-path", action="append", default=[], help="Optional workspace-relative file/directory for pytest or compileall; unittest accepts one discovery directory; may be repeated")
    code_run.add_argument("--timeout", type=int, default=120, help="Per-sandbox test timeout in seconds (1..600)")
    code_inspect = code_sub.add_parser("inspect", help="Inspect the persistent status, diff, and verification evidence for a coding run")
    code_inspect.add_argument("run_id")
    sandbox = sub.add_parser("sandbox", help="Inspect the local coding-test isolation backend")
    sandbox_sub = sandbox.add_subparsers(dest="sandbox_command", required=True)
    sandbox_sub.add_parser("status", help="Check local Docker daemon and digest-pinned sandbox image; runs no user code")
    sandbox_cleanup = sandbox_sub.add_parser("cleanup", help="Force-remove only containers recorded for one interrupted Ayu coding run")
    sandbox_cleanup.add_argument("run_id")
    news = sub.add_parser("news", help="Optional NewsAPI.org metadata search; no AI provider is called")
    news_sub = news.add_subparsers(dest="news_command", required=True)
    news_search = news_sub.add_parser("search", help="Search article metadata with NewsAPI.org")
    news_search.add_argument("query", nargs="+", help="Search terms")
    news_search.add_argument("--from-date")
    news_search.add_argument("--to-date")
    news_search.add_argument("--sources", help="Comma-separated NewsAPI source IDs")
    news_search.add_argument("--sort-by", choices=("relevancy", "popularity", "publishedAt"), default="publishedAt")
    news_search.add_argument("--limit", type=int, default=10)
    news_search.add_argument("--allow-network", action="store_true", required=True, help="Explicitly authorize this external network request")
    self_test = sub.add_parser("test", help="Run bounded offline checks without touching user data")
    self_test.add_argument("--ayulm-smoke", action="store_true", help="Also perform one tiny CPU AyuLM training/checkpoint/generation test; requires optional PyTorch")
    backup=sub.add_parser("backup",help="Create a private local SQLite backup")
    backup.add_argument("output",type=Path)
    restore=sub.add_parser("restore",help="Restore the local database (destructive; explicit --confirm required)")
    restore.add_argument("input",type=Path); restore.add_argument("--confirm",action="store_true",help="Confirm replacing the configured local database")
    docs = sub.add_parser("docs", help="Search local text documents; does not use a model")
    ds = docs.add_subparsers(dest="docs_command", required=True)
    dsearch = ds.add_parser("search", help="Search TXT, Markdown, source, JSON, and CSV files under a root")
    dsearch.add_argument("root", type=Path); dsearch.add_argument("query"); dsearch.add_argument("--limit", type=int, default=20)
    train = sub.add_parser("train", help="Train the native AyuLM decoder on a user-provided text corpus (requires optional PyTorch)")
    train.add_argument("--data", required=True, type=Path); train.add_argument("--output", type=Path, default=Path("ayu-checkpoint.pt")); train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seq-len", type=int, default=64); train.add_argument("--d-model", type=int, default=64); train.add_argument("--heads", type=int, default=4); train.add_argument("--layers", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=4); train.add_argument("--eval-interval", type=int, default=20); train.add_argument("--seed", type=int, default=42); train.add_argument("--device", default="cpu")
    resume_train = sub.add_parser("train-resume", help="Resume native AyuLM training from a compatible checkpoint")
    resume_train.add_argument("checkpoint", type=Path); resume_train.add_argument("--data", required=True, type=Path); resume_train.add_argument("--steps", type=int, default=100); resume_train.add_argument("--device", default="cpu")
    evaluate = sub.add_parser("evaluate", help="Evaluate a native AyuLM checkpoint on a text corpus")
    evaluate.add_argument("checkpoint", type=Path); evaluate.add_argument("--data", required=True, type=Path); evaluate.add_argument("--device", default="cpu")
    generate = sub.add_parser("generate", help="Generate text with an AyuLM-native checkpoint (not a pretrained assistant)")
    generate.add_argument("checkpoint", type=Path); generate.add_argument("prompt"); generate.add_argument("--tokens", type=int, default=80); generate.add_argument("--device", default="cpu")
    benchmark = sub.add_parser("benchmark", help="Measure AyuLM autoregressive generation speed on this machine (not a quality score)")
    benchmark.add_argument("checkpoint", type=Path); benchmark.add_argument("--tokens", type=int, default=32); benchmark.add_argument("--repeats", type=int, default=3); benchmark.add_argument("--device", default="cpu")
    checkpoint = sub.add_parser("checkpoint", help="Inspect and validate an AyuLM checkpoint")
    checkpoint.add_subparsers(dest="checkpoint_command", required=True).add_parser("inspect", help="Read checkpoint metadata and verify weight/config compatibility").add_argument("path", type=Path)
    return parser


def lexical_context(store: Store, prompt: str, project_id: str | None = None) -> str:
    words = {w.lower() for w in re.findall(r"[\w'-]{3,}", prompt)} - {"the","and","for","that","this","with","from","have","what","when","where","your","you","are"}
    if not words: return ""
    candidates = store.recall(limit=100, project_id=project_id)
    ranked = []
    for row in candidates:
        tokens = {w.lower() for w in re.findall(r"[\w'-]{3,}", row["content"])}
        overlap = len(words & tokens)
        if overlap: ranked.append((overlap / max(1, len(tokens)), row))
    ranked.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
    selected = [row for _, row in ranked[:5]]
    if not selected: return ""
    return "User-saved local memories relevant by lexical overlap (not independently verified):\n" + "\n".join(f"- [{r['category']}] {r['content']}" for r in selected)


def run_chat(store: Store, settings: Settings, args: argparse.Namespace) -> int:
    mode = args.mode or settings.mode
    if mode == "basic":
        print("Basic mode provides local CLI, memory, project, task, and document-search features; it does not perform LLM inference. Use --mode local with a verified model, or explicitly --mode remote with provider credentials.", file=sys.stderr)
        return 2
    provider = build_local_provider(settings) if mode == "local" else build_remote_provider(settings)
    session_id = args.session or uuid.uuid4().hex[:12]
    print(f"Ayu AI session: {session_id} (mode: {mode}; role: {args.role})")
    if mode == "local":
        print("Local inference only; no model is downloaded and no remote fallback occurs.")
    print("Type /exit to leave; /help shows commands.")
    def ask(text: str) -> bool:
        if not text.strip(): return True
        store.add_message(session_id, "user", text)
        msgs = [{"role":"system","content":ROLES[args.role]}]
        context = lexical_context(store, text) if settings.memory_enabled else ""
        if context: msgs.append({"role":"system","content":context})
        msgs.extend(store.history(session_id))
        if isinstance(provider,LlamaServerProvider):
            pieces=[]
            print("Ayu: ",end="",flush=True)
            stream=provider.stream_complete(msgs)
            try:
                for piece in stream:
                    pieces.append(piece)
                    print(piece,end="",flush=True)
            except KeyboardInterrupt:
                print("\nAyu response cancelled; the partial response was not saved.")
                return False
            except ProviderError as exc:
                print(f"\nAyu error: {exc}",file=sys.stderr); return False
            finally:
                stream.close()
            answer="".join(pieces).strip()
            if not answer:
                print("\nAyu error: local server stream returned no text.",file=sys.stderr); return False
            print("\n")
        else:
            try: answer = provider.complete(msgs)
            except ProviderError as exc:
                print(f"Ayu error: {exc}", file=sys.stderr); return False
            print(f"\n{answer}\n")
        store.add_message(session_id, "assistant", answer)
        return True
    if args.once is not None: return 0 if ask(args.once) else 1
    while True:
        try: line = input("you> ").strip()
        except (EOFError,KeyboardInterrupt): print("\nSession ended."); return 0
        if line in {"/exit","/quit"}: return 0
        if line == "/help": print("/exit, /quit — end chat; /help — help. Durable memory is saved explicitly with `ayu memory add`."); continue
        ask(line)


def _project(store: Store, command: str, args: argparse.Namespace) -> int:
    if command == "create":
        if args.root and not Path(args.root).expanduser().is_dir(): print("Project root must be an existing directory.", file=sys.stderr); return 2
        try: pid = store.create_project(args.name, args.root)
        except Exception as exc: print(f"Could not create project: {exc}", file=sys.stderr); return 2
        print(f"Created project {args.name!r} (ID: {pid})."); return 0
    if command == "list":
        rows = store.list_projects()
        for p in rows: print(f"{p['id']}\t{p['name']}\t{p['root_path'] or '(no root)'}")
        if not rows: print("No projects. Create one with `ayu project create NAME`.")
        return 0
    p = store.get_project(args.project)
    if not p: print(f"Project not found: {args.project}", file=sys.stderr); return 1
    try: state = json.loads(p["state_json"])
    except (json.JSONDecodeError, TypeError): state = {}
    tasks = store.list_tasks(p["id"])
    if command == "inspect":
        print(f"Project: {p['name']} ({p['id']})\nRoot: {p['root_path'] or '(none)'}\nUpdated: {p['updated_at']}\nSaved state: {json.dumps(state,ensure_ascii=False)}")
        for t in tasks:
            dependencies=store.task_dependencies(t["id"])
            suffix=f" (depends on {', '.join('#'+str(x) for x in dependencies)})" if dependencies else ""
            print(f"Task #{t['id']} [{t['status']}] {t['title']}{suffix}")
        return 0
    if command == "update":
        try: value=json.loads(args.value)
        except json.JSONDecodeError as exc: print(f"--value must be valid JSON: {exc}",file=sys.stderr); return 2
        if args.field=="current_state" and not isinstance(value,str): print("current_state must be a JSON string.",file=sys.stderr); return 2
        if args.field!="current_state" and (not isinstance(value,list) or any(not isinstance(item,str) for item in value)):
            print(f"{args.field} must be a JSON array of strings.",file=sys.stderr); return 2
        state[args.field]=value
        if store.set_project_state(p["id"],state): print(f"Updated project {p['id']} field {args.field}."); return 0
        print("Project update failed.",file=sys.stderr); return 1
    root_ok = bool(p["root_path"] and Path(p["root_path"]).is_dir()) if p["root_path"] else None
    print(f"Resume check: {p['name']}\nReferenced root: {'exists' if root_ok else 'missing' if root_ok is False else 'not configured'}")
    artifacts=state.get("artifact_references",[])
    artifacts_ok=True
    if isinstance(artifacts,list):
        for ref in artifacts:
            candidate=Path(str(ref)).expanduser()
            if not candidate.is_absolute() and p["root_path"]: candidate=Path(p["root_path"])/candidate
            exists=candidate.exists(); artifacts_ok=artifacts_ok and exists
            print(f"Artifact {ref}: {'exists' if exists else 'missing'}")
    ready = store.ready_tasks(p["id"])
    pending = [t for t in tasks if t["status"] in {"pending","in_progress"}]
    print(f"Next ready task: #{ready[0]['id']} {ready[0]['title']}" if ready else "Next ready task: none")
    blocked=[t for t in pending if t["id"] not in {r["id"] for r in ready}]
    for task in blocked:
        deps=store.task_dependencies(task["id"])
        print(f"Blocked task #{task['id']} prerequisites: "+", ".join(f"#{dep}" for dep in deps if (store.get_task(dep) or {}).get("status")!="completed"))
    if root_ok is False or not artifacts_ok: return 1
    return 0


def run_ayulm(args: argparse.Namespace) -> int:
    from . import ayulm
    def progress(item):
        print(f"step {item['step']}: train_loss={item.get('train_loss')} validation_loss={item.get('validation_loss')}",file=sys.stderr)
    try:
        if args.command == "train":
            config = ayulm.ModelConfig(seq_len=args.seq_len,d_model=args.d_model,n_heads=args.heads,n_layers=args.layers)
            result=ayulm.train(args.data,args.output,args.steps,config,seed=args.seed,eval_interval=args.eval_interval,batch_size=args.batch_size,device=args.device,progress=progress)
            print(json.dumps(result,indent=2)); return 0
        if args.command == "train-resume":
            model,meta=ayulm.load_checkpoint(args.checkpoint,args.device)
            config=ayulm.ModelConfig(**meta["model_config"])
            seed=int(meta.get("seed",42))
            del model,meta
            result=ayulm.train(args.data,args.checkpoint,args.steps,config,seed=seed,resume=args.checkpoint,device=args.device,progress=progress)
            print(json.dumps(result,indent=2)); return 0
        if args.command == "checkpoint":
            model,meta=ayulm.load_checkpoint(args.path)
            print(json.dumps({"architecture":meta["architecture"],"architecture_version":meta["architecture_version"],"model_config":meta["model_config"],"parameter_count":ayulm.parameter_count(model),"tokenizer_id":meta["tokenizer_id"],"step":meta.get("step"),"metrics":meta.get("metrics"),"training_data_path":meta.get("training_data_path"),"checkpoint":str(args.path.resolve())},indent=2)); return 0
        if args.command == "generate":
            if not 1 <= args.tokens <= 2048: raise ValueError("--tokens must be between 1 and 2048")
            import torch
            model,meta=ayulm.load_checkpoint(args.checkpoint,args.device)
            tokenizer=ayulm.ByteTokenizer(); prefix=tokenizer.encode(args.prompt,add_bos=True)
            if len(prefix)>model.config.seq_len: prefix=prefix[-model.config.seq_len:]
            output=model.generate(torch.tensor([prefix],dtype=torch.long,device=args.device),max_new_tokens=args.tokens)
            print(tokenizer.decode(output[0].tolist()[len(prefix):])); return 0
        if args.command == "benchmark":
            import platform, statistics, time, torch
            if not 1 <= args.tokens <= 2048: raise ValueError("--tokens must be between 1 and 2048")
            if not 1 <= args.repeats <= 10: raise ValueError("--repeats must be between 1 and 10")
            model,meta=ayulm.load_checkpoint(args.checkpoint,args.device)
            prefix=ayulm.ByteTokenizer().encode("Ayu benchmark",add_bos=True)[-model.config.seq_len:]
            initial=torch.tensor([prefix],dtype=torch.long,device=args.device)
            torch.manual_seed(0)
            model.generate(initial,max_new_tokens=min(2,args.tokens))
            def synchronize():
                if str(args.device).startswith("cuda"): torch.cuda.synchronize(args.device)
            measurements=[]
            for _ in range(args.repeats):
                synchronize(); started=time.perf_counter()
                model.generate(initial.clone(),max_new_tokens=args.tokens)
                synchronize(); measurements.append(time.perf_counter()-started)
            total=sum(measurements)
            print(json.dumps({"checkpoint":str(args.checkpoint.resolve()),"parameters":ayulm.parameter_count(model),"device":str(args.device),"torch_version":torch.__version__,"python":platform.python_version(),"prompt_tokens":len(prefix),"generated_tokens_per_repeat":args.tokens,"repeats":args.repeats,"seconds_per_repeat":measurements,"median_seconds":statistics.median(measurements),"mean_tokens_per_second":args.tokens*args.repeats/total,"note":"Local autoregressive throughput only; not a quality score or cross-machine comparison."},indent=2)); return 0
        if args.command == "evaluate":
            import mmap, torch
            model,meta=ayulm.load_checkpoint(args.checkpoint,args.device)
            data_path=args.data.expanduser().resolve()
            if data_path.stat().st_size<model.config.seq_len+1: raise ValueError(f"evaluation data needs at least {model.config.seq_len+1} bytes")
            total=0.0; count=0; model.eval()
            with data_path.open("rb") as source, mmap.mmap(source.fileno(),0,access=mmap.ACCESS_READ) as raw, torch.no_grad():
                for start in range(0,len(raw)-model.config.seq_len,model.config.seq_len):
                    x=torch.tensor([[b+4 for b in raw[start:start+model.config.seq_len]]],dtype=torch.long,device=args.device)
                    y=torch.tensor([[b+4 for b in raw[start+1:start+model.config.seq_len+1]]],dtype=torch.long,device=args.device)
                    _,loss=model(x,y)
                    if not torch.isfinite(loss): raise ValueError("non-finite evaluation loss")
                    total+=float(loss.item())*y.numel(); count+=y.numel()
            if not count: raise ValueError("evaluation corpus produced no complete sequence")
            loss=total/count; print(json.dumps({"data":str(data_path),"bytes":data_path.stat().st_size,"loss":loss,"perplexity":ayulm._perplexity(loss),"note":"Byte-level next-token validation only; not conversational quality."},indent=2)); return 0
    except (RuntimeError,ValueError,OSError,KeyError,TypeError) as exc:
        print(f"AyuLM error: {exc}",file=sys.stderr); return 1


def run_agent(store: Store, settings: Settings, args: argparse.Namespace) -> int:
    from .agents import AgentRunError, AgentRunner
    from .tools import ToolError, ToolRegistry
    if args.agents_command == "roles":
        print("Sequential roles: planner -> bounded read-only researcher/tools -> analyst -> reviewer. Each model call uses the explicitly selected provider; there is no parallel or recursive loop.")
        return 0
    if args.agents_command == "inspect":
        run = store.get_agent_run(args.run_id)
        if run is None:
            print("Agent run not found.", file=sys.stderr)
            return 1
        events = store.list_agent_events(args.run_id)
        run["events"] = []
        for event in events:
            payload = json.loads(event.pop("payload_json"))
            run["events"].append({**event, "payload": payload})
        print(json.dumps(run, ensure_ascii=False, indent=2))
        return 0
    mode = args.mode or settings.mode
    if mode not in {"local", "remote"}:
        print("Agent execution requires explicit local or remote inference; basic mode does not invent model reasoning.", file=sys.stderr)
        return 2
    if mode == "remote" and not args.allow_remote_context:
        print("Remote agent mode may send the task and retrieved workspace excerpts to the provider. Re-run with --allow-remote-context only if you explicitly consent.", file=sys.stderr)
        return 2
    try:
        workspace = args.workspace.expanduser().resolve(strict=True)
        if not workspace.is_dir(): raise ToolError("--workspace must be an existing directory.")
        project_id = None
        if args.project:
            project = store.get_project(args.project)
            if project is None: raise ToolError(f"Saved project not found: {args.project}")
            if not project["root_path"]: raise ToolError("Selected project has no saved root; configure one before associating an agent run.")
            project_root = Path(project["root_path"]).expanduser().resolve(strict=True)
            if not project_root.is_dir() or not workspace.is_relative_to(project_root):
                raise ToolError("Approved workspace must be inside the selected project's existing root.")
            project_id = project["id"]
        if not 1 <= args.max_steps <= 4: raise ToolError("--max-steps must be between 1 and 4.")
        tools = ToolRegistry(workspace)
        provider = build_local_provider(settings) if mode == "local" else build_remote_provider(settings)
        runner = AgentRunner(provider, store, tools, mode=mode)
        result = runner.run(" ".join(args.task), max_steps=args.max_steps, project_id=project_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (AgentRunError, ToolError, OSError, ValueError) as exc:
        print(f"Agent run failed: {exc}", file=sys.stderr)
        return 1


def run_coding(store: Store, settings: Settings, args: argparse.Namespace) -> int:
    from .coding import CodeRunError, CodingRunner
    from .sandbox import DockerSandboxRunner, SandboxError
    if args.code_command == "inspect":
        run = store.get_agent_run(args.run_id)
        if run is None:
            print("Coding run not found.", file=sys.stderr); return 1
        events = []
        for event in store.list_agent_events(args.run_id):
            payload = json.loads(event.pop("payload_json"))
            events.append({**event, "payload": payload})
        print(json.dumps({"run": run, "events": events}, ensure_ascii=False, indent=2))
        return 0
    if settings.mode != "local":
        print("Coding runs require AYU_MODE=local; remote coding-context transfer is not enabled in this initial implementation.", file=sys.stderr)
        return 2
    if settings.local_backend != "llama-server":
        print("Coding runs require AYU_LOCAL_BACKEND=llama-server for strict local JSON Schema output; llama-cli remains available for ordinary local chat.", file=sys.stderr)
        return 2
    sandbox = DockerSandboxRunner.from_environment()
    try:
        policy = sandbox.preflight()
    except SandboxError as exc:
        print(f"Coding run refused: required Docker sandbox is unavailable: {exc}", file=sys.stderr)
        return 2
    print("Execution boundary: local Docker container; network disabled; non-root; resource-bounded; disposable workspace snapshot.", file=sys.stderr)
    try:
        provider = build_local_provider(settings)
        runner = CodingRunner(provider, store, mode="local", sandbox_runner=sandbox)
        result = runner.run(" ".join(args.task), args.workspace, suite=args.test_suite,
                            test_paths=args.test_path, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1
    except (CodeRunError, ProviderError, OSError, ValueError, SandboxError) as exc:
        print(f"Coding run failed: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(); return 0
    if args.command == "test":
        from .selftest import run_smoke_test
        return run_smoke_test(include_ayulm=args.ayulm_smoke)
    if args.command in {"train","train-resume","evaluate","generate","benchmark","checkpoint"}:
        return run_ayulm(args)
    if args.command == "init":
        target = Path.cwd() / ".env"
        if target.exists() and not args.force: print(f"Configuration already exists at {target}; not overwriting. Use --force to replace it."); return 1
        try:
            target.write_text("# Ayu AI local configuration. No API key is required for basic/local mode.\nAYU_MODE=local\nAYU_MEMORY_ENABLED=true\n# AYU_LOCAL_MODEL=/absolute/path/to/model.gguf\n# Preferred chat-template-aware backend: start llama-server separately, bound to 127.0.0.1.\n# AYU_LOCAL_BACKEND=llama-server\n# AYU_LLAMA_SERVER_URL=http://127.0.0.1:9931\n# AYU_LOCAL_MODEL_ID=optional-exact-id-from-v1-models\n# Legacy one-shot CLI backend: AYU_LOCAL_BACKEND=llama-cli and AYU_LLAMA_CLI=llama-cli\n# AYU_LOCAL_MAX_INPUT_CHARS=65536\n# AYU_DATABASE=~/.ayu-ai/ayu.sqlite3\n# Select one optional remote inference provider explicitly:\n# AYU_MODE=remote\n# AYU_PROVIDER=openai\n# OPENAI_API_KEY=placeholder-only\n# AYU_MODEL=your-openai-model-id\n# AYU_PROVIDER=anthropic\n# ANTHROPIC_API_KEY=placeholder-only\n# ANTHROPIC_MODEL=your-anthropic-model-id\n# For OpenAI-compatible services use AYU_PROVIDER=openai-compatible, AYU_API_KEY, AYU_BASE_URL, AYU_MODEL.\n# Optional NewsAPI requires NEWSAPI_API_KEY and AYU_NEWSAPI_ENABLED=true; searches require --allow-network.\n",encoding="utf-8")
            try: target.chmod(0o600)
            except OSError: pass
        except OSError as exc:
            print(f"Could not create configuration: {exc}",file=sys.stderr); return 1
        print(f"Created {target}. No credentials or models were downloaded."); return 0
    try:
        settings = Settings.load()
        store = Store(settings.database_path)
    except (ValueError,OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr); return 2
    cmd = args.command
    if cmd == "code": return run_coding(store, settings, args)
    if cmd == "sandbox":
        from .sandbox import DockerSandboxRunner
        runner = DockerSandboxRunner.from_environment()
        if args.sandbox_command == "status":
            report = runner.status()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready"] else 2
        run = store.get_agent_run(args.run_id)
        if run is None:
            print("Coding run not found.", file=sys.stderr)
            return 1
        names = []
        for item in store.list_agent_events(args.run_id):
            if item.get("event_type") != "sandbox_container_starting":
                continue
            try:
                payload = json.loads(item["payload_json"])
                name = payload.get("container_name")
                if isinstance(name, str) and name not in names:
                    names.append(name)
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        if not names:
            print("No audited sandbox container was recorded for this run.", file=sys.stderr)
            return 1
        try:
            outcomes = [{"container_name": name, "removed_or_absent": runner.cleanup_container(name)} for name in names]
        except SandboxError as exc:
            print(f"Sandbox cleanup failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"run_id": args.run_id, "cleanup": outcomes}, ensure_ascii=False, indent=2))
        return 0 if all(item["removed_or_absent"] for item in outcomes) else 1
    if cmd == "doctor":
        parent = settings.database_path.parent
        remote_selected=settings.mode=="remote"
        endpoint_ok = settings.remote_provider == "anthropic" or remote_endpoint_error(settings.base_url) is None
        remote_ready=bool(settings.api_key and settings.model) and endpoint_ok
        if settings.local_backend == "llama-server":
            runtime_check=("llama-server endpoint (not contacted)",True,settings.llama_server_url)
        else:
            runtime_check=("llama.cpp executable",bool(shutil.which(settings.llama_cli)) or Path(settings.llama_cli).is_file(),settings.llama_cli)
        checks = [("Python",sys.version_info >= (3,10),sys.version.split()[0]),("Database directory",parent.exists() and parent.is_dir(),str(parent)),("Local model",not settings.local_model_path or settings.local_model_path.is_file(),str(settings.local_model_path or "not configured")),runtime_check,("Remote configuration",remote_ready if remote_selected else True,"ready" if remote_ready else "incomplete/unsafe; optional unless AYU_MODE=remote")]
        for label,ok,detail in checks: print(f"{'INFO' if label.startswith('llama-server endpoint') or label=='llama.cpp executable' else 'OK' if ok else 'NEEDS ATTENTION'}  {label}: {detail}")
        return 0 if all(ok or label in {"llama.cpp executable","llama-server endpoint (not contacted)"} for label,ok,_ in checks) else 1
    if cmd == "status":
        messages,memories=store.counts(); print(f"Ayu AI {__version__}\nMode: {settings.mode}\nDatabase: {settings.database_path}\nPersistent memory enabled: {settings.memory_enabled}\nLocal backend: {settings.local_backend}\nLocal server URL: {settings.llama_server_url if settings.local_backend == 'llama-server' else 'not selected'}\nConfigured local model: {settings.local_model_path or 'none'}\nRemote provider: {settings.remote_provider}\nRemote model configured: {'yes' if settings.model else 'no'}\nSelected AI credential: {'present (value hidden)' if settings.api_key else 'not configured'}\nNewsAPI enabled: {settings.newsapi_enabled}\nNewsAPI credential: {'present (value hidden)' if settings.newsapi_api_key else 'not configured'}\nMessages: {messages}\nMemories: {memories}\nNo telemetry is sent."); return 0
    if cmd == "chat": return run_chat(store,settings,args)
    if cmd == "models":
        if args.models_command == "list":
            print(f"Local GGUF: {settings.local_model_path or '(not configured)'}\nRemote provider: {settings.remote_provider}\nRemote: {'configured (opt-in only)' if settings.model and settings.api_key else '(not configured)'}")
            registry=Path.home()/".ayu-ai"/"model-registry.json"
            try: entries=json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []
            except (OSError,json.JSONDecodeError): entries=[]
            entries=entries if isinstance(entries,list) else []
            if entries:
                print("AyuLM native research checkpoints:")
                for entry in entries:
                    if isinstance(entry,dict):
                        cp=entry.get("checkpoint_path")
                        available=isinstance(cp,str) and Path(cp).is_file()
                        print(f"  {entry.get('model_id','(unknown)')} | {entry.get('parameter_count','?')} parameters | {'available' if available else 'missing'}")
            else: print("AyuLM native checkpoints: none registered; see `ayu train --help` (optional PyTorch).")
            return 0
        if args.models_command == "info":
            p=settings.local_model_path
            if settings.local_backend == "llama-server":
                runtime=f"llama-server at {settings.llama_server_url} (loopback only)"
            else:
                runtime=f"llama.cpp CLI: {settings.llama_cli}"
            print(f"Local backend: {settings.local_backend}\nRuntime: {runtime}\nLocal model: {p or 'not configured'}\nModel file present: {bool(p and p.is_file())}\nRemote provider: {settings.remote_provider}\nRemote mode configured: {bool(settings.api_key and settings.model)}\nRemote calls are never used in local mode."); return 0
        try:
            provider=build_local_provider(settings)
            if isinstance(provider,LlamaServerProvider):
                model_id=provider.validate()
                runtime=provider.last_runtime_info
                detail=f"Local llama-server is ready; model `{model_id}`, build {runtime.get('build_info', 'not reported')}, effective context {runtime.get('context_size', 'unknown')} tokens, and loaded model path matches AYU_LOCAL_MODEL."
            else:
                model,exe=provider.validate()
                detail=f"GGUF file and executable preflight passed: {model} ({model.stat().st_size:,} bytes); runtime {exe}."
            if args.infer:
                answer=provider.complete([{"role":"system","content":"This is a local inference smoke test. Respond briefly."},{"role":"user","content":"Return the text AYU_LOCAL_INFERENCE_OK."}])
                selected=(f"llama-server model {provider.last_runtime_info.get('model_id')} build {provider.last_runtime_info.get('build_info')}" if isinstance(provider,LlamaServerProvider) else f"llama-cli runtime {exe}")
                print(f"Local inference request succeeded; a non-empty response was received from {selected}.\n{answer}")
            else:
                print(f"Preflight passed; this did not run model generation. {detail}\nRun `ayu models verify --infer` to test actual inference.")
            return 0
        except ProviderError as exc: print(f"Model verification failed: {exc}",file=sys.stderr); return 1
    if cmd == "memory":
        subcmd=args.memory_command
        if subcmd=="status":
            _,count=store.counts(); print(f"Memories: {count}\nEnabled for retrieval/writes: {settings.memory_enabled}\nDatabase: {settings.database_path}\nCategories: {', '.join(sorted(MEMORY_CATEGORIES))}\nMemories are local; use `ayu memory delete ID` or remove the database to erase them."); return 0
        if subcmd=="add":
            if not settings.memory_enabled: print("Persistent memory is disabled (AYU_MEMORY_ENABLED=false).",file=sys.stderr); return 2
            try: mid=store.remember(" ".join(args.text),args.category,args.project)
            except ValueError as exc: print(exc,file=sys.stderr); return 2
            print(f"Saved memory #{mid} [{args.category}]."); return 0
        if subcmd=="search":
            rows=store.recall(" ".join(args.query),category=args.category,project_id=args.project)
            for r in rows: print(f"#{r['id']} [{r['category']}] {r['created_at']} {r['content']}")
            if not rows: print("No matching memories.")
            return 0
        if subcmd=="show":
            row=store.get_memory(args.id)
            if not row: print("Memory not found.",file=sys.stderr); return 1
            print(f"#{row['id']} [{row['category']}] project={row['project_id']} source={row['provenance']} created={row['created_at']}\n{row['content']}"); return 0
        if subcmd=="update":
            if not settings.memory_enabled: print("Persistent memory is disabled (AYU_MEMORY_ENABLED=false).",file=sys.stderr); return 2
            if args.text is None and args.category is None: print("Specify --text and/or --category.",file=sys.stderr); return 2
            try: changed=store.update_memory(args.id,args.text,args.category)
            except ValueError as exc: print(exc,file=sys.stderr); return 2
            print("Memory updated." if changed else "Memory not found."); return 0 if changed else 1
        if subcmd=="export":
            target=args.output.expanduser().resolve()
            if target==settings.database_path.resolve() or target in {Path(str(settings.database_path.resolve())+"-wal"),Path(str(settings.database_path.resolve())+"-shm")}:
                print("Export destination cannot be the active database or one of its SQLite sidecars.",file=sys.stderr); return 2
            if target.exists() and not args.force: print(f"File exists: {target}; use --force to overwrite.",file=sys.stderr); return 2
            target.parent.mkdir(parents=True,exist_ok=True)
            records=[dict(row) for row in store.all_memories()]
            fd,tmp=tempfile.mkstemp(prefix="ayu-memory-",suffix=".tmp",dir=target.parent)
            try:
                with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump({"format":"ayu-memory-export-v1","memories":records},stream,ensure_ascii=False,indent=2); stream.write("\n")
                os.chmod(tmp,0o600)
                if target.exists() and not args.force: os.unlink(tmp); print(f"File exists: {target}; use --force to overwrite.",file=sys.stderr); return 2
                os.replace(tmp,target)
            except (OSError,TypeError) as exc:
                try: os.unlink(tmp)
                except OSError: pass
                print(f"Memory export failed: {exc}",file=sys.stderr); return 1
            print(f"Exported {len(records)} memories to {target}."); return 0
        if subcmd=="import":
            if not settings.memory_enabled: print("Persistent memory is disabled (AYU_MEMORY_ENABLED=false).",file=sys.stderr); return 2
            try:
                import_path=args.input.expanduser()
                if import_path.stat().st_size>25*1024*1024: raise ValueError("memory export exceeds the 25 MB safety limit")
                payload=json.loads(import_path.read_text(encoding="utf-8"))
                if not isinstance(payload,dict) or payload.get("format")!="ayu-memory-export-v1" or not isinstance(payload.get("memories"),list): raise ValueError("unsupported memory export format")
                count=store.import_memories(payload["memories"])
            except (OSError,UnicodeError,json.JSONDecodeError,ValueError) as exc: print(f"Memory import failed: {exc}",file=sys.stderr); return 1
            print(f"Imported or merged {count} memories."); return 0
        removed=store.forget(args.id); print("Memory deleted." if removed else "Memory not found."); return 0 if removed else 1
    if cmd == "project": return _project(store,args.project_command,args)
    if cmd == "tasks":
        if args.tasks_command=="add":
            p=store.get_project(args.project)
            if not p: print("Project not found.",file=sys.stderr); return 1
            try: tid=store.add_task(p["id"],args.title,args.depends_on)
            except Exception as exc: print(f"Could not add task: {exc}",file=sys.stderr); return 2
            print(f"Added task #{tid}."); return 0
        if args.tasks_command=="list":
            p=store.get_project(args.project)
            if not p: print("Project not found.",file=sys.stderr); return 1
            rows=store.list_tasks(p["id"])
            for t in rows:
                dependencies=store.task_dependencies(t["id"])
                suffix=f" (depends on {', '.join('#'+str(x) for x in dependencies)})" if dependencies else ""
                print(f"#{t['id']} [{t['status']}] {t['title']}{suffix}")
            if not rows: print("No tasks.")
            return 0
        if args.tasks_command=="inspect":
            task=store.get_task(args.id)
            if task is None: print("Task not found.",file=sys.stderr); return 1
            dependencies=store.task_dependencies(args.id)
            unmet=[dep for dep in dependencies if (store.get_task(dep) or {}).get("status")!="completed"]
            print(json.dumps({"task":task,"depends_on":dependencies,"ready":not unmet and task["status"] in {"pending","in_progress"},"unmet_dependencies":unmet},indent=2))
            return 0
        if args.tasks_command=="set-dependencies":
            try: store.set_task_dependencies(args.id,args.depends_on)
            except ValueError as exc: print(f"Could not update task dependencies: {exc}",file=sys.stderr); return 2
            print(f"Dependencies updated for task #{args.id}."); return 0
        status={"start":"in_progress","complete":"completed","cancel":"cancelled"}[args.tasks_command]
        try: ok=store.update_task_status(args.id,status)
        except ValueError as exc: print(f"Task status change rejected: {exc}",file=sys.stderr); return 2
        label={"start":"Task started","complete":"Task completed","cancel":"Task cancelled"}[args.tasks_command]
        print(label+"." if ok else "Task not found."); return 0 if ok else 1
    if cmd == "config":
        if args.config_command=="show":
            print(f"AYU_MODE={settings.mode}\nAYU_MEMORY_ENABLED={str(settings.memory_enabled).lower()}\nAYU_DATABASE={settings.database_path}\nAYU_LOCAL_BACKEND={settings.local_backend}\nAYU_LOCAL_MODEL={settings.local_model_path or ''}\nAYU_LOCAL_MODEL_ID={settings.local_model_id or ''}\nAYU_LLAMA_CLI={settings.llama_cli}\nAYU_LLAMA_SERVER_URL={settings.llama_server_url}\nAYU_LOCAL_MAX_INPUT_CHARS={settings.local_max_input_chars}\nAYU_PROVIDER={settings.remote_provider}\nAYU_BASE_URL={settings.base_url}\nAYU_MODEL={settings.model or ''}\nSelected AI credential={'<redacted: set>' if settings.api_key else '<not set>'}\nNEWSAPI_API_KEY={'<redacted: set>' if settings.newsapi_api_key else '<not set>'}\nAYU_NEWSAPI_ENABLED={str(settings.newsapi_enabled).lower()}"); return 0
        if settings.mode=="remote":
            problems=[]
            key_name={"openai":"OPENAI_API_KEY","anthropic":"ANTHROPIC_API_KEY"}.get(settings.remote_provider,"AYU_API_KEY")
            model_name={"openai":"AYU_MODEL or OPENAI_MODEL","anthropic":"AYU_MODEL or ANTHROPIC_MODEL"}.get(settings.remote_provider,"AYU_MODEL")
            if not settings.api_key: problems.append(f"{key_name} is missing")
            if not settings.model: problems.append(f"{model_name} is missing")
            endpoint_error=None if settings.remote_provider=="anthropic" else remote_endpoint_error(settings.base_url)
            if endpoint_error: problems.append(endpoint_error)
            if problems: print("Remote configuration invalid: "+"; ".join(problems),file=sys.stderr); return 1
        print("Configuration parses successfully. Remote settings are not used in the selected mode; this check sends no network requests."); return 0
    if cmd == "tools":
        from .tools import ToolRegistry
        with tempfile.TemporaryDirectory(prefix="ayu-tool-list-") as temporary:
            metadata = ToolRegistry(Path(temporary)).list_tools()
        print(json.dumps({"tools":metadata,"policy":"Read-only; explicit workspace required; no shell, file modification, or network tools."},indent=2))
        return 0
    if cmd == "agents": return run_agent(store, settings, args)
    if cmd == "news":
        if settings.mode == "basic":
            print("Online research is disabled in basic/offline mode.", file=sys.stderr); return 2
        if not args.allow_network:
            print("NewsAPI makes an external request; re-run this command with --allow-network to authorize it.",file=sys.stderr); return 2
        try:
            result=NewsAPIClient(settings).search(" ".join(args.query),from_date=args.from_date,to_date=args.to_date,sources=args.sources,sort_by=args.sort_by,page_size=args.limit)
            print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
        except ProviderError as exc:
            print(f"NewsAPI search failed: {exc}",file=sys.stderr); return 1
    if cmd == "backup":
        target=args.output.expanduser().resolve()
        if target.exists(): print(f"Backup destination already exists: {target}",file=sys.stderr); return 2
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix="ayu-db-backup-",suffix=".sqlite3",dir=target.parent); os.close(fd)
        try:
            with contextlib.closing(sqlite3.connect(settings.database_path)) as source, contextlib.closing(sqlite3.connect(tmp)) as dest:
                source.backup(dest)
                result=dest.execute("PRAGMA integrity_check").fetchone()[0]
                if result!="ok": raise sqlite3.DatabaseError(f"backup integrity check failed: {result}")
            os.chmod(tmp,0o600)
            if target.exists(): raise FileExistsError(f"Backup destination already exists: {target}")
            os.replace(tmp,target)
        except (OSError,sqlite3.DatabaseError) as exc:
            try: os.unlink(tmp)
            except OSError: pass
            print(f"Backup failed: {exc}",file=sys.stderr); return 1
        print(f"Verified database backup created at {target}."); return 0
    if cmd == "restore":
        source_path=args.input.expanduser().resolve(); destination=settings.database_path.resolve()
        if not args.confirm: print("Restore not performed. Re-run with --confirm after backing up the current database.",file=sys.stderr); return 2
        if not source_path.is_file() or source_path==destination: print("Restore source must be an existing backup file distinct from the active database.",file=sys.stderr); return 2
        fd,staged=tempfile.mkstemp(prefix="ayu-restore-",suffix=".sqlite3",dir=destination.parent); os.close(fd)
        rollback=destination.with_name(destination.name+".pre-restore-"+uuid.uuid4().hex+".sqlite3")
        staged_live=False
        try:
            with contextlib.closing(sqlite3.connect(source_path)) as source:
                source.execute("PRAGMA query_only=ON")
                if source.execute("PRAGMA integrity_check").fetchone()[0]!="ok": raise sqlite3.DatabaseError("source integrity check failed")
                tables={r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"messages","memories"}.issubset(tables): raise sqlite3.DatabaseError("source is not an Ayu AI database")
                with contextlib.closing(sqlite3.connect(staged)) as target:
                    source.backup(target)
                    target.execute("PRAGMA journal_mode=DELETE")
                    if target.execute("PRAGMA integrity_check").fetchone()[0]!="ok": raise sqlite3.DatabaseError("staged restore integrity check failed")
            os.chmod(staged,0o600)
            # Preserve the current database before touching it; the backup uses SQLite's
            # consistent online-copy API rather than copying a live file byte-for-byte.
            if destination.exists():
                fd,old_copy=tempfile.mkstemp(prefix="ayu-before-restore-",suffix=".sqlite3",dir=destination.parent); os.close(fd)
                try:
                    with contextlib.closing(sqlite3.connect(destination)) as old_db, contextlib.closing(sqlite3.connect(old_copy)) as old_backup:
                        old_db.backup(old_backup)
                        if old_backup.execute("PRAGMA integrity_check").fetchone()[0]!="ok": raise sqlite3.DatabaseError("pre-restore backup integrity check failed")
                        old_backup.execute("PRAGMA journal_mode=DELETE")
                    os.chmod(old_copy,0o600); os.replace(old_copy,rollback)
                except BaseException:
                    try: os.unlink(old_copy)
                    except OSError: pass
                    raise
                with contextlib.closing(sqlite3.connect(destination)) as live:
                    checkpoint=live.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if checkpoint and checkpoint[0]!=0: raise sqlite3.DatabaseError("active database is busy; close other Ayu processes and retry")
                    live.execute("PRAGMA journal_mode=DELETE")
            for suffix in ("-wal","-shm"):
                try: Path(str(destination)+suffix).unlink(missing_ok=True)
                except OSError as exc: raise sqlite3.DatabaseError(f"could not clear SQLite sidecar {suffix}: {exc}") from exc
            os.replace(staged,destination); staged_live=True
            Store(destination)  # Apply additive schema upgrades after restoration.
        except (OSError,sqlite3.DatabaseError,ValueError) as exc:
            if staged_live and rollback.exists():
                try:
                    for suffix in ("-wal","-shm"): Path(str(destination)+suffix).unlink(missing_ok=True)
                    os.replace(rollback,destination); Store(destination)
                except (OSError,sqlite3.DatabaseError,ValueError) as recovery_error:
                    print(f"CRITICAL: restore failed ({exc}); automatic rollback also failed ({recovery_error}). Preserve {rollback} and inspect the database before further use.",file=sys.stderr); return 1
            print(f"Restore failed before completion: {exc}"+(f". Current-database backup: {rollback}" if rollback.exists() else ". The active database was not replaced."),file=sys.stderr); return 1
        finally:
            try: Path(staged).unlink(missing_ok=True)
            except OSError: pass
        print(f"Restored Ayu AI database from {source_path}. Pre-restore recovery copy: {rollback}"); return 0
    if cmd == "docs":
        root=args.root.expanduser().resolve()
        if not root.is_dir(): print("Document search root must be an existing directory.",file=sys.stderr); return 2
        if not args.query.strip(): print("Search query cannot be empty.",file=sys.stderr); return 2
        exts={".txt",".md",".rst",".py",".js",".ts",".json",".csv",".toml",".yaml",".yml",".html",".css",".sh",".ps1"}
        found=0; files_seen=0
        try:
            for path in root.rglob("*"):
                if found>=max(1,min(args.limit,100)): break
                files_seen+=1
                if files_seen>5000: print("Stopped after the 5,000-file safety limit."); break
                if not path.is_file() or path.suffix.lower() not in exts: continue
                resolved=path.resolve()
                if root not in resolved.parents: continue
                try:
                    if resolved.stat().st_size>5*1024*1024: continue
                    with resolved.open("r",encoding="utf-8") as stream:
                        for n,line in enumerate(stream,1):
                            if args.query.casefold() in line.casefold():
                                print(f"{resolved}:{n}:{line[:500]}"); found+=1
                                if found>=max(1,min(args.limit,100)): break
                except (OSError,UnicodeError): continue
        except OSError as exc: print(f"Search error: {exc}",file=sys.stderr); return 1
        if not found: print("No matches. Search is local and lexical; no model or external service was used.")
        return 0
    parser.print_help(); return 0

if __name__ == "__main__": raise SystemExit(main())
