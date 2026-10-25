"""Terminal-first Ayu AI CLI. Basic commands work without any AI credentials."""
from __future__ import annotations
import argparse, contextlib, json, os, re, shutil, sqlite3, sys, tempfile, uuid
from pathlib import Path
from . import __version__
from .settings import Settings
from .provider import ChatProvider, LlamaCppProvider, ProviderError, remote_endpoint_error
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
    ms.add_parser("verify", help="Verify the configured local GGUF file and llama.cpp executable")
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
    ta.add_argument("project"); ta.add_argument("title")
    tl = ts.add_parser("list", help="List project tasks")
    tl.add_argument("project")
    tc = ts.add_parser("cancel", help="Cancel a pending task")
    tc.add_argument("id", type=int)
    config = sub.add_parser("config", help="Inspect and validate local configuration")
    cs = config.add_subparsers(dest="config_command", required=True)
    cs.add_parser("show", help="Show configuration (credential value is redacted)")
    cs.add_parser("validate", help="Validate configuration without contacting providers")
    tools = sub.add_parser("tools", help="Show implemented tool capabilities")
    tools.add_subparsers(dest="tools_command", required=True).add_parser("list", help="List available local tools")
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
    provider = LlamaCppProvider(settings) if mode == "local" else ChatProvider(settings)
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
        try: answer = provider.complete(msgs)
        except ProviderError as exc:
            print(f"Ayu error: {exc}", file=sys.stderr); return False
        store.add_message(session_id, "assistant", answer)
        print(f"\n{answer}\n"); return True
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
        for t in tasks: print(f"Task #{t['id']} [{t['status']}] {t['title']}")
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
    pending = [t for t in tasks if t["status"] in {"pending","in_progress"}]
    print(f"Next pending task: #{pending[0]['id']} {pending[0]['title']}" if pending else "Next pending task: none")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(); return 0
    if args.command in {"train","train-resume","evaluate","generate","checkpoint"}:
        return run_ayulm(args)
    if args.command == "init":
        target = Path.cwd() / ".env"
        if target.exists() and not args.force: print(f"Configuration already exists at {target}; not overwriting. Use --force to replace it."); return 1
        try:
            target.write_text("# Ayu AI local configuration. No API key is required for basic/local mode.\nAYU_MODE=local\n# AYU_LOCAL_MODEL=/absolute/path/to/model.gguf\n# AYU_LLAMA_CLI=llama-cli\n# AYU_DATABASE=~/.ayu-ai/ayu.sqlite3\n# Remote mode is opt-in: set AYU_MODE=remote, AYU_API_KEY, AYU_MODEL, and AYU_BASE_URL.\n",encoding="utf-8")
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
    if cmd == "doctor":
        parent = settings.database_path.parent
        remote_selected=settings.mode=="remote"
        remote_ready=bool(settings.api_key and settings.model) and remote_endpoint_error(settings.base_url) is None
        checks = [("Python",sys.version_info >= (3,10),sys.version.split()[0]),("Database directory",parent.exists() and parent.is_dir(),str(parent)),("Local model",not settings.local_model_path or settings.local_model_path.is_file(),str(settings.local_model_path or "not configured")),("llama.cpp",bool(shutil.which(settings.llama_cli)) or Path(settings.llama_cli).is_file(),settings.llama_cli),("Remote configuration",remote_ready if remote_selected else True,"ready" if remote_ready else "incomplete/unsafe; optional unless AYU_MODE=remote")]
        for label,ok,detail in checks: print(f"{'OK' if ok else 'INFO' if label in {'llama.cpp','Remote credentials'} else 'NEEDS ATTENTION'}  {label}: {detail}")
        return 0 if all(ok or label=="llama.cpp" for label,ok,_ in checks) else 1
    if cmd == "status":
        messages,memories=store.counts(); print(f"Ayu AI {__version__}\nMode: {settings.mode}\nDatabase: {settings.database_path}\nPersistent memory enabled: {settings.memory_enabled}\nConfigured local model: {settings.local_model_path or 'none'}\nRemote model configured: {'yes' if settings.model else 'no'}\nAPI key: {'present (value hidden)' if settings.api_key else 'not configured'}\nMessages: {messages}\nMemories: {memories}\nNo telemetry is sent."); return 0
    if cmd == "chat": return run_chat(store,settings,args)
    if cmd == "models":
        if args.models_command == "list":
            print(f"Local GGUF: {settings.local_model_path or '(not configured)'}\nRemote: {'configured (opt-in only)' if settings.model and settings.api_key else '(not configured)'}")
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
            print(f"Local backend: llama.cpp CLI\nRuntime: {settings.llama_cli}\nLocal model: {p or 'not configured'}\nModel present: {bool(p and p.is_file())}\nRemote mode configured: {bool(settings.api_key and settings.model)}\nRemote calls are never used in local mode."); return 0
        try:
            p,exe=LlamaCppProvider(settings).validate()
            print(f"Verified local GGUF: {p}\nSize: {p.stat().st_size:,} bytes\nRuntime: {exe}\nNo license is inferred; review the model source's license separately."); return 0
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
            try: tid=store.add_task(p["id"],args.title)
            except Exception as exc: print(f"Could not add task: {exc}",file=sys.stderr); return 2
            print(f"Added task #{tid}."); return 0
        if args.tasks_command=="list":
            p=store.get_project(args.project)
            if not p: print("Project not found.",file=sys.stderr); return 1
            rows=store.list_tasks(p["id"])
            for t in rows: print(f"#{t['id']} [{t['status']}] {t['title']}")
            if not rows: print("No tasks.")
            return 0
        ok=store.update_task_status(args.id,"cancelled"); print("Task cancelled." if ok else "Task not found."); return 0 if ok else 1
    if cmd == "config":
        if args.config_command=="show":
            print(f"AYU_MODE={settings.mode}\nAYU_MEMORY_ENABLED={str(settings.memory_enabled).lower()}\nAYU_DATABASE={settings.database_path}\nAYU_LOCAL_MODEL={settings.local_model_path or ''}\nAYU_LLAMA_CLI={settings.llama_cli}\nAYU_BASE_URL={settings.base_url}\nAYU_MODEL={settings.model or ''}\nAYU_API_KEY={'<redacted: set>' if settings.api_key else '<not set>'}"); return 0
        if settings.mode=="remote":
            problems=[]
            if not settings.api_key: problems.append("AYU_API_KEY is missing")
            if not settings.model: problems.append("AYU_MODEL is missing")
            endpoint_error=remote_endpoint_error(settings.base_url)
            if endpoint_error: problems.append(endpoint_error)
            if problems: print("Remote configuration invalid: "+"; ".join(problems),file=sys.stderr); return 1
        print("Configuration parses successfully. Remote settings are not used in the selected mode; this check sends no network requests."); return 0
    if cmd == "tools":
        print("Implemented local tools:\n- SQLite conversation and memory storage\n- project/task bookkeeping\n- lexical search of explicitly supplied local text documents\nNot implemented: shell execution, file modification, browser access, autonomous code execution, external research."); return 0
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
