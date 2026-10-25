"""Bounded, read-only local tools for model-assisted repository analysis.

No tool in this first registry executes commands, writes files, accesses the
network, or escapes the explicitly approved workspace.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".py", ".pyi", ".toml", ".ini", ".cfg",
    ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".css",
    ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c",
    ".h", ".cpp", ".hpp", ".sh", ".ps1", ".sql", ".xml", ".log",
}
DENIED_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", "secrets.json"}
DENIED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db"}
DENIED_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}


class ToolError(ValueError):
    """A tool request was rejected or could not be safely completed."""


@dataclass(frozen=True)
class ToolDefinition:
    tool_id: str
    purpose: str
    input_schema: dict[str, Any]
    required_permissions: tuple[str, ...]
    timeout_seconds: float
    side_effect: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def public_metadata(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "purpose": self.purpose,
            "input_schema": self.input_schema,
            "required_permissions": list(self.required_permissions),
            "timeout_seconds": self.timeout_seconds,
            "side_effect": self.side_effect,
        }


class ToolRegistry:
    """A fixed allow-list of bounded read-only tools under one workspace."""

    def __init__(self, workspace: Path, *, max_files: int = 200, max_total_bytes: int = 5 * 1024 * 1024):
        try:
            root = workspace.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ToolError(f"Workspace cannot be resolved: {exc}") from exc
        if not root.is_dir():
            raise ToolError("Workspace must be an existing directory.")
        if max_files < 1 or max_files > 1000 or max_total_bytes < 1024 or max_total_bytes > 50 * 1024 * 1024:
            raise ToolError("Tool resource limits are outside the supported safety bounds.")
        self.workspace = root
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self._tools = {
            "search_text": ToolDefinition(
                tool_id="search_text",
                purpose="Search supported text files under the approved workspace for a literal query.",
                input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 160}}, "additionalProperties": False},
                required_permissions=("workspace.read",), timeout_seconds=3.0, side_effect="read_only",
                handler=self._search_text,
            ),
            "read_file": ToolDefinition(
                tool_id="read_file",
                purpose="Read a bounded line range from a supported UTF-8 text file inside the approved workspace.",
                input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 512}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}, "additionalProperties": False},
                required_permissions=("workspace.read",), timeout_seconds=3.0, side_effect="read_only",
                handler=self._read_file,
            ),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.public_metadata() for tool in self._tools.values()]

    def invoke(self, tool_id: str, arguments: Any) -> dict[str, Any]:
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ToolError(f"Tool is not allow-listed: {tool_id!r}")
        self._validate_arguments(tool, arguments)
        started = time.monotonic()
        result = tool.handler(arguments)
        if time.monotonic() - started > tool.timeout_seconds:
            raise ToolError(f"Tool {tool_id!r} exceeded its {tool.timeout_seconds:g}s time budget; result discarded.")
        return result

    @staticmethod
    def _validate_arguments(tool: ToolDefinition, arguments: Any) -> None:
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object.")
        schema = tool.input_schema
        required = set(schema["required"])
        allowed = set(schema["properties"])
        if required - arguments.keys():
            raise ToolError(f"Missing required tool arguments: {', '.join(sorted(required - arguments.keys()))}")
        if arguments.keys() - allowed:
            raise ToolError(f"Unexpected tool arguments: {', '.join(sorted(arguments.keys() - allowed))}")
        for key, value in arguments.items():
            spec = schema["properties"][key]
            if spec["type"] == "string":
                if not isinstance(value, str) or not spec["minLength"] <= len(value) <= spec["maxLength"]:
                    raise ToolError(f"{key} must be a string of length {spec['minLength']}..{spec['maxLength']}.")
            elif spec["type"] == "integer":
                if isinstance(value, bool) or not isinstance(value, int) or value < spec["minimum"]:
                    raise ToolError(f"{key} must be an integer >= {spec['minimum']}.")

    def _safe_file(self, raw_path: str) -> Path:
        # Use POSIX-style relative paths in tool requests, on every host OS.
        if "\\" in raw_path or PureWindowsPath(raw_path).is_absolute():
            raise ToolError("Tool paths must be relative POSIX paths inside the approved workspace.")
        relative = Path(raw_path)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} or part.startswith(".") for part in relative.parts):
            raise ToolError("Hidden, empty, absolute, and parent-traversal paths are not allowed.")
        current = self.workspace
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ToolError("Symbolic links are not allowed in tool paths.")
        try:
            resolved = current.resolve(strict=True)
        except OSError as exc:
            raise ToolError(f"File cannot be resolved: {exc}") from exc
        if not resolved.is_relative_to(self.workspace):
            raise ToolError("Resolved path escapes the approved workspace.")
        if not resolved.is_file():
            raise ToolError("Tool path must name a regular file.")
        if resolved.name.casefold() in DENIED_NAMES or resolved.suffix.casefold() in DENIED_SUFFIXES:
            raise ToolError("This file type/name is blocked from model-assisted reads.")
        if resolved.suffix.casefold() not in TEXT_SUFFIXES:
            raise ToolError("File extension is not in the supported text-only allow-list.")
        if not os.access(resolved, os.R_OK):
            raise ToolError("File is not readable by this process.")
        size = resolved.stat().st_size
        if size > 512 * 1024:
            raise ToolError("File exceeds the 512 KiB per-file read limit.")
        return resolved

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_file(arguments["path"])
        try:
            with path.open("rb") as stream: data=stream.read(512*1024+1)
            if len(data)>512*1024: raise ToolError("File exceeds the 512 KiB per-file read limit.")
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("File is not valid UTF-8 text; binary or undecodable content is not passed to the model.") from exc
        except OSError as exc:
            raise ToolError(f"Could not read file: {exc}") from exc
        lines = text.splitlines()
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line", min(len(lines), start + 149) if lines else start)
        if end < start or end - start >= 150:
            raise ToolError("Requested line range must be ordered and no longer than 150 lines.")
        if start > len(lines) + 1:
            raise ToolError(f"start_line exceeds file length ({len(lines)} lines).")
        return {"path": path.relative_to(self.workspace).as_posix(), "start_line": start, "end_line": min(end, len(lines)), "total_lines": len(lines), "content": "\n".join(lines[start - 1:end])[:20_000]}

    def _search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"].casefold()
        deadline = time.monotonic() + 3.0
        results: list[dict[str, Any]] = []
        visited_files = 0
        visited_directories = 0
        bytes_read = 0
        truncated = False
        pending=[self.workspace]
        while pending:
            directory=pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if time.monotonic()>deadline or visited_files>=self.max_files or visited_directories>=self.max_files or bytes_read>=self.max_total_bytes or len(results)>=20:
                            truncated=True; break
                        name=entry.name
                        if entry.is_symlink() or name.startswith("."): continue
                        if entry.is_dir(follow_symlinks=False):
                            if name.casefold() in DENIED_DIRS: continue
                            visited_directories+=1
                            if visited_directories>self.max_files: truncated=True; break
                            pending.append(Path(entry.path)); continue
                        path=Path(entry.path)
                        if not entry.is_file(follow_symlinks=False) or name.casefold() in DENIED_NAMES or path.suffix.casefold() in DENIED_SUFFIXES or path.suffix.casefold() not in TEXT_SUFFIXES:
                            continue
                        visited_files+=1
                        try:
                            size=entry.stat(follow_symlinks=False).st_size
                            if size>512*1024 or bytes_read+size>self.max_total_bytes or not os.access(path,os.R_OK): continue
                            with path.open("rb") as stream: data=stream.read(512*1024+1)
                            if len(data)>512*1024: continue
                            content=data.decode("utf-8")
                        except (OSError,UnicodeError): continue
                        bytes_read+=len(data)
                        for line_number,line in enumerate(content.splitlines(),1):
                            if line_number%64==0 and time.monotonic()>deadline:
                                truncated=True; break
                            if query in line.casefold():
                                results.append({"path":path.relative_to(self.workspace).as_posix(),"line":line_number,"snippet":line[:500]})
                                if len(results)>=20: truncated=True; break
                        if truncated: break
            except OSError:
                continue
            if truncated: break
        return {"query": arguments["query"], "matches": results, "files_visited": visited_files, "bytes_read": bytes_read, "truncated": truncated}
