"""SQLite persistence for chat, categorized local memories, projects, and tasks."""
from __future__ import annotations
import json, os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

MEMORY_CATEGORIES = {"working", "episodic", "semantic", "project", "preference"}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink(): raise OSError(f"SQLite database path must not be a symbolic link: {self.path}")
        for suffix in ("-wal","-shm"):
            if Path(str(self.path)+suffix).is_symlink(): raise OSError(f"SQLite sidecar must not be a symbolic link: {self.path}{suffix}")
        try:
            fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
            os.close(fd)
        except FileExistsError: pass
        self._initialize()

    @contextmanager
    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
            self._secure_permissions()

    def _secure_permissions(self) -> None:
        if os.name == "nt": return
        for path in (self.path,Path(str(self.path)+"-wal"),Path(str(self.path)+"-shm")):
            try:
                if path.exists() and not path.is_symlink(): path.chmod(0o600)
            except OSError: pass

    def _initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id,id);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, root_path TEXT,
                    state_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','in_progress','completed','cancelled','failed')),
                    notes TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    depends_on_task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    PRIMARY KEY(task_id,depends_on_task_id),
                    CHECK(task_id != depends_on_task_id));
                CREATE INDEX IF NOT EXISTS idx_task_dependencies_target ON task_dependencies(depends_on_task_id);
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY, task TEXT NOT NULL,
                    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('local','remote')),
                    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','cancelled','unknown_outcome')),
                    max_steps INTEGER NOT NULL CHECK(max_steps BETWEEN 1 AND 4),
                    error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at DESC);
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0), actor TEXT NOT NULL,
                    event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(run_id,sequence));
                CREATE INDEX IF NOT EXISTS idx_agent_events_run ON agent_events(run_id,sequence);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
            if "category" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'semantic'")
            if "project_id" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN project_id TEXT")
            if "provenance" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT 'user'")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(category,project_id,id)")
        self._migrate_agent_run_unknown_outcome()

    def _migrate_agent_run_unknown_outcome(self) -> None:
        """Rebuild the constrained run table once, preserving rows and child events."""
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='agent_runs'").fetchone()
            if row is None or "unknown_outcome" in (row[0] or ""):
                db.commit()
                return
            db.execute("""CREATE TABLE agent_runs_new (
                id TEXT PRIMARY KEY, task TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                mode TEXT NOT NULL CHECK(mode IN ('local','remote')),
                status TEXT NOT NULL CHECK(status IN ('running','completed','failed','cancelled','unknown_outcome')),
                max_steps INTEGER NOT NULL CHECK(max_steps BETWEEN 1 AND 4),
                error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            db.execute("INSERT INTO agent_runs_new SELECT id,task,project_id,mode,status,max_steps,error,created_at,updated_at FROM agent_runs")
            db.execute("DROP TABLE agent_runs")
            db.execute("ALTER TABLE agent_runs_new RENAME TO agent_runs")
            db.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at DESC)")
            violations = db.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError("agent run migration would leave broken foreign keys")
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
            self._secure_permissions()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        with self.connect() as db:
            db.execute("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)", (session_id, role, content, utc_now()))

    def history(self, session_id: str, limit: int = 20) -> list[dict[str,str]]:
        with self.connect() as db:
            rows = db.execute("SELECT role,content FROM (SELECT id,role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id", (session_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def remember(self, content: str, category: str = "semantic", project_id: str | None = None, provenance: str = "user") -> int:
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(sorted(MEMORY_CATEGORIES))}")
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        with self.connect() as db:
            if project_id and not db.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone():
                raise ValueError(f"project not found: {project_id}")
            existing = db.execute("SELECT id FROM memories WHERE category=? AND ifnull(project_id,'')=ifnull(?,'') AND lower(trim(content))=lower(trim(?)) ORDER BY id DESC LIMIT 1", (category, project_id, content)).fetchone()
            if existing:
                db.execute("UPDATE memories SET created_at=?,provenance=? WHERE id=?", (utc_now(), provenance, existing["id"]))
                return int(existing["id"])
            cur = db.execute("INSERT INTO memories(content,created_at,category,project_id,provenance) VALUES(?,?,?,?,?)", (content, utc_now(), category, project_id, provenance))
            return int(cur.lastrowid)

    def recall(self, query: str = "", limit: int = 10, category: str | None = None, project_id: str | None = None) -> list[sqlite3.Row]:
        if category and category not in MEMORY_CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        clauses, params = [], []
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("content LIKE ? ESCAPE '\\'"); params.append(f"%{escaped}%")
        if category:
            clauses.append("category=?"); params.append(category)
        if project_id:
            clauses.append("project_id=?"); params.append(project_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            return db.execute(f"SELECT id,content,created_at,category,project_id,provenance FROM memories{where} ORDER BY id DESC LIMIT ?", (*params, max(1,min(limit,100)))).fetchall()

    def update_memory(self, memory_id: int, content: str | None = None, category: str | None = None) -> bool:
        if category is not None and category not in MEMORY_CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        with self.connect() as db:
            row = db.execute("SELECT content,category FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row: return False
            next_content = content.strip() if content is not None else row["content"]
            next_category = category or row["category"]
            if not next_content: raise ValueError("memory content cannot be empty")
            db.execute("UPDATE memories SET content=?,category=?,created_at=? WHERE id=?", (next_content,next_category,utc_now(),memory_id))
            return True

    def forget(self, memory_id: int) -> bool:
        with self.connect() as db: return db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount > 0

    def create_project(self, name: str, root_path: str | None = None) -> str:
        name = name.strip()
        if not name: raise ValueError("project name cannot be empty")
        project_id = name.lower().replace(" ", "-")
        now = utc_now()
        initial_state = {"requirements":[],"goals":[],"architecture_decisions":[],"current_state":"","completed":[],"pending":[],"known_problems":[],"checkpoints":[],"artifact_references":[]}
        with self.connect() as db:
            db.execute("INSERT INTO projects(id,name,root_path,state_json,created_at,updated_at) VALUES(?,?,?,?,?,?)", (project_id,name,str(Path(root_path).expanduser().resolve()) if root_path else None,json.dumps(initial_state),now,now))
        return project_id

    def list_projects(self) -> list[sqlite3.Row]:
        with self.connect() as db: return db.execute("SELECT id,name,root_path,created_at,updated_at FROM projects ORDER BY name").fetchall()

    def get_project(self, project_id: str) -> sqlite3.Row | None:
        with self.connect() as db: return db.execute("SELECT * FROM projects WHERE id=? OR name=?", (project_id,project_id)).fetchone()

    def set_project_state(self, project_id: str, state: dict) -> bool:
        with self.connect() as db:
            return db.execute("UPDATE projects SET state_json=?,updated_at=? WHERE id=?", (json.dumps(state,ensure_ascii=False),utc_now(),project_id)).rowcount > 0

    def get_memory(self, memory_id: int) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute("SELECT id,content,created_at,category,project_id,provenance FROM memories WHERE id=?",(memory_id,)).fetchone()

    def all_memories(self) -> list[sqlite3.Row]:
        with self.connect() as db:
            return db.execute("SELECT id,content,created_at,category,project_id,provenance FROM memories ORDER BY id").fetchall()

    def import_memories(self, rows: list[dict]) -> int:
        """Validate all rows before inserting; a failure rolls back the import."""
        prepared=[]
        for row in rows:
            if not isinstance(row,dict) or not isinstance(row.get("content"),str) or not row["content"].strip():
                raise ValueError("each import record must contain non-empty string content")
            category=row.get("category","semantic")
            if category not in MEMORY_CATEGORIES: raise ValueError(f"invalid memory category: {category}")
            project_id=row.get("project_id")
            if project_id is not None and not isinstance(project_id,str): raise ValueError("project_id must be a string or null")
            created_at=row.get("created_at")
            if created_at is not None and not isinstance(created_at,str): raise ValueError("created_at must be a string")
            prepared.append((row["content"].strip(),category,project_id,created_at or utc_now()))
        with self.connect() as db:
            for _,_,project_id,_ in prepared:
                if project_id and not db.execute("SELECT 1 FROM projects WHERE id=?",(project_id,)).fetchone():
                    raise ValueError(f"import references missing project: {project_id}")
            for content,category,project_id,created_at in prepared:
                existing=db.execute("SELECT id FROM memories WHERE category=? AND ifnull(project_id,'')=ifnull(?,'') AND lower(trim(content))=lower(trim(?))",(category,project_id,content)).fetchone()
                if existing:
                    db.execute("UPDATE memories SET created_at=?,provenance='imported' WHERE id=?",(created_at,existing["id"]))
                else:
                    db.execute("INSERT INTO memories(content,created_at,category,project_id,provenance) VALUES(?,?,?,?, 'imported')",(content,created_at,category,project_id))
        return len(prepared)

    def add_task(self, project_id: str, title: str, depends_on: list[int] | None = None) -> int:
        if not isinstance(title,str): raise ValueError("task title must be a string")
        title=title.strip()
        if not title: raise ValueError("task title cannot be empty")
        dependencies=self._normalize_dependencies([] if depends_on is None else depends_on)
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM projects WHERE id=?",(project_id,)).fetchone(): raise ValueError(f"project not found: {project_id}")
            self._validate_dependency_targets(db,project_id,dependencies)
            cur = db.execute("INSERT INTO tasks(project_id,title,created_at,updated_at) VALUES(?,?,?,?)", (project_id,title,utc_now(),utc_now()))
            task_id=int(cur.lastrowid)
            db.executemany("INSERT INTO task_dependencies(task_id,depends_on_task_id) VALUES(?,?)",[(task_id,dep) for dep in dependencies])
            return task_id

    @staticmethod
    def _normalize_dependencies(dependencies: list[int]) -> list[int]:
        if not isinstance(dependencies,list) or len(dependencies)>128: raise ValueError("dependencies must be a list of at most 128 task IDs")
        if any(isinstance(item,bool) or not isinstance(item,int) or item<=0 for item in dependencies): raise ValueError("dependency IDs must be positive integers")
        if len(set(dependencies))!=len(dependencies): raise ValueError("dependency IDs must not be repeated")
        return dependencies

    @staticmethod
    def _validate_dependency_targets(db: sqlite3.Connection, project_id: str, dependencies: list[int]) -> None:
        if not dependencies: return
        placeholders=",".join("?" for _ in dependencies)
        rows=db.execute(f"SELECT id,project_id FROM tasks WHERE id IN ({placeholders})",dependencies).fetchall()
        if len(rows)!=len(dependencies): raise ValueError("one or more dependency tasks do not exist")
        if any(row["project_id"]!=project_id for row in rows): raise ValueError("task dependencies must belong to the same project")

    def set_task_dependencies(self, task_id: int, dependencies: list[int]) -> None:
        dependencies=self._normalize_dependencies(dependencies)
        if isinstance(task_id,bool) or not isinstance(task_id,int) or task_id<=0: raise ValueError("task ID must be a positive integer")
        with self.connect() as db:
            task=db.execute("SELECT project_id,status FROM tasks WHERE id=?",(task_id,)).fetchone()
            if not task: raise ValueError(f"task not found: {task_id}")
            if task["status"] in {"in_progress","completed"}: raise ValueError("dependencies cannot be changed after a task starts")
            self._validate_dependency_targets(db,task["project_id"],dependencies)
            graph: dict[int,list[int]]={}
            for row in db.execute("SELECT task_id,depends_on_task_id FROM task_dependencies"):
                graph.setdefault(int(row["task_id"]),[]).append(int(row["depends_on_task_id"]))
            graph[task_id]=dependencies
            pending=list(dependencies); visited=set()
            while pending:
                current=pending.pop()
                if current==task_id: raise ValueError("task dependency would create a cycle")
                if current in visited: continue
                visited.add(current); pending.extend(graph.get(current,[]))
            db.execute("DELETE FROM task_dependencies WHERE task_id=?",(task_id,))
            db.executemany("INSERT INTO task_dependencies(task_id,depends_on_task_id) VALUES(?,?)",[(task_id,dep) for dep in dependencies])

    def list_tasks(self, project_id: str) -> list[sqlite3.Row]:
        with self.connect() as db: return db.execute("SELECT * FROM tasks WHERE project_id=? ORDER BY id", (project_id,)).fetchall()

    def get_task(self, task_id: int):
        with self.connect() as db:
            row=db.execute("SELECT * FROM tasks WHERE id=?",(task_id,)).fetchone()
        return dict(row) if row else None

    def task_dependencies(self, task_id: int) -> list[int]:
        with self.connect() as db:
            rows=db.execute("SELECT depends_on_task_id FROM task_dependencies WHERE task_id=? ORDER BY depends_on_task_id",(task_id,)).fetchall()
        return [int(row[0]) for row in rows]

    def ready_tasks(self, project_id: str) -> list[sqlite3.Row]:
        ready=[]
        with self.connect() as db:
            rows=db.execute("SELECT t.* FROM tasks t WHERE t.project_id=? AND t.status IN ('pending','in_progress') AND NOT EXISTS (SELECT 1 FROM task_dependencies d JOIN tasks p ON p.id=d.depends_on_task_id WHERE d.task_id=t.id AND p.status!='completed') ORDER BY t.id",(project_id,)).fetchall()
        return rows

    def update_task_status(self, task_id: int, status: str) -> bool:
        if not isinstance(status,str) or status not in {"pending","in_progress","completed","cancelled","failed"}: raise ValueError("invalid task status")
        if isinstance(task_id,bool) or not isinstance(task_id,int) or task_id<=0: raise ValueError("task ID must be a positive integer")
        with self.connect() as db:
            task=db.execute("SELECT id,status FROM tasks WHERE id=?",(task_id,)).fetchone()
            if not task: return False
            if status in {"in_progress","completed"}:
                blocked=db.execute("SELECT p.id FROM task_dependencies d JOIN tasks p ON p.id=d.depends_on_task_id WHERE d.task_id=? AND p.status!='completed' ORDER BY p.id",(task_id,)).fetchall()
                if blocked: raise ValueError("task is blocked by incomplete prerequisites: "+", ".join(str(row["id"]) for row in blocked))
            if task["status"]=="completed" and status!="completed":
                raise ValueError("completed tasks cannot be reopened; create a new task to preserve dependency history")
            return db.execute("UPDATE tasks SET status=?,updated_at=? WHERE id=?", (status,utc_now(),task_id)).rowcount > 0

    def counts(self) -> tuple[int,int]:
        with self.connect() as db:
            return (int(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]), int(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]))

    def create_agent_run(self, run_id: str, task: str, mode: str, max_steps: int,
                         project_id: str | None = None,
                         initial_event: tuple[str, str, dict] | None = None) -> None:
        if not isinstance(run_id,str) or not run_id or len(run_id) > 64: raise ValueError("invalid agent run ID")
        if not isinstance(task,str): raise ValueError("agent task must be text")
        task = task.strip()
        if not task or len(task) > 2000: raise ValueError("agent task must be 1..2000 characters")
        if mode not in {"local", "remote"}: raise ValueError("agent mode must be local or remote")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 4:
            raise ValueError("agent max_steps must be 1..4")
        now = utc_now()
        serialized_event = None
        if initial_event is not None:
            if (not isinstance(initial_event, tuple) or len(initial_event) != 3
                    or not isinstance(initial_event[0], str) or not initial_event[0] or len(initial_event[0]) > 48
                    or not isinstance(initial_event[1], str) or not initial_event[1] or len(initial_event[1]) > 48
                    or not isinstance(initial_event[2], dict)):
                raise ValueError("invalid initial agent event")
            serialized_event = json.dumps(initial_event[2], ensure_ascii=False, separators=(",", ":"))
            if len(serialized_event.encode("utf-8")) > 64 * 1024:
                raise ValueError("initial agent event exceeds the 64 KiB audit limit")
        with self.connect() as db:
            db.execute("INSERT INTO agent_runs(id,task,project_id,mode,status,max_steps,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (run_id,task,project_id,mode,"running",max_steps,now,now))
            if serialized_event is not None:
                db.execute("INSERT INTO agent_events(run_id,sequence,actor,event_type,payload_json,created_at) VALUES(?,0,?,?,?,?)",
                           (run_id, initial_event[0], initial_event[1], serialized_event, now))

    def record_agent_event(self, run_id: str, sequence: int, actor: str, event_type: str, payload: dict) -> None:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0: raise ValueError("event sequence must be a non-negative integer")
        if not isinstance(actor,str) or not actor or len(actor) > 48 or not isinstance(event_type,str) or not event_type or len(event_type) > 48: raise ValueError("invalid agent event actor/type")
        if not isinstance(payload,dict): raise ValueError("agent event payload must be a JSON object")
        serialized=json.dumps(payload,ensure_ascii=False,separators=(",",":"))
        if len(serialized.encode("utf-8")) > 64 * 1024: raise ValueError("agent event exceeds the 64 KiB audit limit")
        with self.connect() as db:
            db.execute("INSERT INTO agent_events(run_id,sequence,actor,event_type,payload_json,created_at) VALUES(?,?,?,?,?,?)", (run_id,sequence,actor,event_type,serialized,utc_now()))
            db.execute("UPDATE agent_runs SET updated_at=? WHERE id=? AND status='running'",(utc_now(),run_id))

    def finish_agent_run(self, run_id: str, status: str, error: str = "") -> bool:
        if status not in {"completed","failed","cancelled"}: raise ValueError("invalid terminal agent run status")
        if not isinstance(error,str): raise ValueError("agent run error must be text")
        with self.connect() as db:
            return db.execute("UPDATE agent_runs SET status=?,error=?,updated_at=? WHERE id=? AND status='running'",(status,error[:1000],utc_now(),run_id)).rowcount == 1

    def mark_agent_run_unknown_outcome(self, run_id: str, *, confirmed_stopped: bool) -> dict[str, object]:
        """Mark a coding run indeterminate after explicit operator confirmation; never retries or cleans resources."""
        if not isinstance(run_id, str) or not run_id or len(run_id) > 64:
            raise ValueError("invalid agent run ID")
        if confirmed_stopped is not True:
            raise ValueError("operator must confirm the Ayu process is stopped before recovery")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM agent_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("agent run not found")
            if row["status"] == "unknown_outcome":
                return {"run_id": run_id, "status": "unknown_outcome", "changed": False}
            if row["status"] != "running":
                raise ValueError(f"only a running run can be recovered; current status is {row['status']}")
            if not db.execute("SELECT 1 FROM agent_events WHERE run_id=? AND actor='coding' LIMIT 1", (run_id,)).fetchone():
                raise ValueError("run has no persisted coding events; refusing to classify it as a coding run")
            sequence = int(db.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM agent_events WHERE run_id=?", (run_id,)).fetchone()[0])
            now = utc_now()
            payload = json.dumps({"previous_status": "running", "status": "unknown_outcome",
                                  "operator_confirmed_process_stopped": True,
                                  "automatic_rerun": False, "worktree_preserved": True,
                                  "container_cleanup_attempted": False}, separators=(",", ":"))
            db.execute("INSERT INTO agent_events(run_id,sequence,actor,event_type,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                       (run_id, sequence, "operator", "recovery_unknown_outcome", payload, now))
            changed = db.execute("UPDATE agent_runs SET status='unknown_outcome',error=?,updated_at=? WHERE id=? AND status='running'",
                                 ("Interrupted run outcome is unknown; inspect audit evidence before any manual action.", now, run_id)).rowcount
            if changed != 1:
                raise sqlite3.IntegrityError("agent run state changed during recovery")
            return {"run_id": run_id, "status": "unknown_outcome", "changed": True}

    def get_agent_run(self, run_id: str):
        with self.connect() as db:
            row=db.execute("SELECT * FROM agent_runs WHERE id=?",(run_id,)).fetchone()
        return dict(row) if row else None

    def list_agent_events(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            rows=db.execute("SELECT sequence,actor,event_type,payload_json,created_at FROM agent_events WHERE run_id=? ORDER BY sequence",(run_id,)).fetchall()
        return [dict(row) for row in rows]
