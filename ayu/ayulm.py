from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProjectManager:
    def __init__(self, projects_path: str | Path | None = None):
        self.projects_path = Path(projects_path) if projects_path else Path.home() / ".ayu" / "projects.json"
        self.projects_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.projects_path.exists():
            self.projects_path.write_text("[]", encoding="utf-8")

    def list_projects(self) -> list[dict[str, Any]]:
        data = json.loads(self.projects_path.read_text(encoding="utf-8") or "[]")
        return data

    def create_project(self, name: str, root: str | Path | None = None, goals: list[str] | None = None) -> dict[str, Any]:
        project_root = Path(root) if root else Path.cwd() / name
        project_root.mkdir(parents=True, exist_ok=True)
        project_file = project_root / ".ayu-project.json"
        payload = {
            "name": name,
            "root": str(project_root),
            "goals": goals or [],
            "status": "initialized",
            "tasks": [],
        }
        project_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        data = self.list_projects()
        data.append(payload)
        self.projects_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return payload

    def inspect_project(self, name: str) -> dict[str, Any]:
        for project in self.list_projects():
            if project["name"] == name:
                return project
        raise FileNotFoundError(f"Project '{name}' not found")

    def resume_project(self, name: str) -> dict[str, Any]:
        project = self.inspect_project(name)
        project["status"] = "resumed"
        return project
