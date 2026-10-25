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
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
            if "category" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'semantic'")
            if "project_id" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN project_id TEXT")
            if "provenance" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT 'user'")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(category,project_id,id)")

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

    def add_task(self, project_id: str, title: str) -> int:
        title=title.strip()
        if not title: raise ValueError("task title cannot be empty")
        with self.connect() as db:
            cur = db.execute("INSERT INTO tasks(project_id,title,created_at,updated_at) VALUES(?,?,?,?)", (project_id,title,utc_now(),utc_now()))
            return int(cur.lastrowid)

    def list_tasks(self, project_id: str) -> list[sqlite3.Row]:
        with self.connect() as db: return db.execute("SELECT * FROM tasks WHERE project_id=? ORDER BY id", (project_id,)).fetchall()

    def update_task_status(self, task_id: int, status: str) -> bool:
        if status not in {"pending","in_progress","completed","cancelled","failed"}: raise ValueError("invalid task status")
        with self.connect() as db: return db.execute("UPDATE tasks SET status=?,updated_at=? WHERE id=?", (status,utc_now(),task_id)).rowcount > 0

    def counts(self) -> tuple[int,int]:
        with self.connect() as db:
            return (int(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]), int(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]))
