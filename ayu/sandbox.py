"""Fail-closed Docker execution boundary for untrusted repository tests.

This module configures a container policy; it does not claim that policy has
been verified on a particular host/runtime. See docs/SECURITY_THREAT_MODEL.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from .tools import DENIED_DIRS, DENIED_NAMES, DENIED_SUFFIXES

MAX_CAPTURE_BYTES = 256 * 1024
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_SNAPSHOT_FILES = 10_000
MAX_SNAPSHOT_ENTRIES = 20_000
SNAPSHOT_TIMEOUT_SECONDS = 60
MAX_FILE_BYTES = 32 * 1024 * 1024
IMAGE_DIGEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}$")
PRIVATE_FILE_NAMES = {
    ".git", ".hg", ".svn", ".netrc", ".npmrc", ".pypirc", ".git-credentials",
    ".gitconfig", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "credentials",
    "credentials.json", "secrets.json", "service-account.json",
}
PRIVATE_SUFFIXES = set(DENIED_SUFFIXES) | {".secret", ".token", ".crt", ".cer", ".der"}
PRIVATE_DIRS = set(DENIED_DIRS) | {
    ".git", ".hg", ".svn", ".ssh", ".aws", ".azure", ".gnupg", ".config",
    ".docker", ".kube", ".terraform", ".npm", ".cache", "__pycache__", "node_modules",
}


class SandboxError(RuntimeError):
    """A required sandbox property is missing, unsupported, or failed."""


@dataclass(frozen=True)
class SandboxPolicy:
    image: str
    memory_bytes: int = 512 * 1024 * 1024
    cpu_count: float = 1.0
    pids_limit: int = 64
    tmpfs_bytes: int = 64 * 1024 * 1024
    uid: int = 65532
    gid: int = 65532
    backend: str = "docker"
    policy_version: str = "ayu-docker-v1"

    def public_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "policy_version": self.policy_version,
            "image": self.image,
            "network": "none",
            "user": f"{self.uid}:{self.gid}",
            "root_filesystem": "read-only",
            "capabilities": "all dropped",
            "no_new_privileges": True,
            "memory_bytes": self.memory_bytes,
            "cpu_count": self.cpu_count,
            "pids_limit": self.pids_limit,
            "tmpfs_bytes": self.tmpfs_bytes,
            "workspace_mount": "read-only disposable task snapshot only",
        }


@dataclass
class SandboxResult:
    command: list[str]
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str
    infrastructure_error: bool
    output_truncated: bool
    backend: str
    image: str
    policy_version: str
    container_name: str
    cleanup_confirmed: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "infrastructure_error": self.infrastructure_error,
            "output_truncated": self.output_truncated,
            "backend": self.backend,
            "image": self.image,
            "policy_version": self.policy_version,
            "container_name": self.container_name,
            "cleanup_confirmed": self.cleanup_confirmed,
        }


class DockerSandboxRunner:
    """Run a fixed Python suite in a no-network, resource-bounded Docker container.

    The image must already exist locally and be addressed by an immutable digest.
    No image pull/build or dependency installation is performed by this runner.
    """

    def __init__(self, image: str | None = None, *, docker_executable: str = "docker",
                 docker_host: str | None = None):
        selected_image = image if image is not None else os.environ.get("AYU_SANDBOX_IMAGE", "").strip()
        executable = docker_executable.strip() if isinstance(docker_executable, str) else ""
        host = docker_host if docker_host is not None else os.environ.get("AYU_DOCKER_HOST", "").strip() or None
        self.image = selected_image
        self.docker_executable = executable
        self.docker_host = host
        self._docker_path: str | None = None
        self.image_env_keys: list[str] = []

    @classmethod
    def from_environment(cls) -> "DockerSandboxRunner":
        return cls(
            image=os.environ.get("AYU_SANDBOX_IMAGE", "").strip(),
            docker_executable=os.environ.get("AYU_DOCKER_BINARY", "docker").strip(),
            docker_host=os.environ.get("AYU_DOCKER_HOST", "").strip() or None,
        )

    @staticmethod
    def _validate_image(image: str) -> None:
        if not isinstance(image, str) or not IMAGE_DIGEST_RE.fullmatch(image):
            raise SandboxError("AYU_SANDBOX_IMAGE must name an immutable image digest (repository@sha256:<64 lowercase hex characters>); tags are refused.")

    @staticmethod
    def _validate_local_host(host: str | None) -> None:
        if host is None:
            return
        if host.startswith("unix://") and os.name != "nt":
            socket_path = host[len("unix://"):]
            if not socket_path.startswith("/") or "\x00" in socket_path:
                raise SandboxError("AYU_DOCKER_HOST must be a local absolute Unix socket or the local Windows named pipe.")
            try:
                if not Path(socket_path).is_socket():
                    raise SandboxError("AYU_DOCKER_HOST must identify an existing Unix socket; remote endpoints are refused.")
            except OSError:
                raise SandboxError("AYU_DOCKER_HOST could not be verified as a Unix socket.") from None
            return
        if os.name == "nt" and host.casefold() in {"npipe:////./pipe/docker_engine", "npipe:////./pipe/docker_engine_linux"}:
            return
        raise SandboxError("Remote Docker daemon endpoints are not allowed; AYU_DOCKER_HOST must be a local Unix socket or Windows named pipe.")

    def _environment(self, docker_config: Path) -> dict[str, str]:
        # Docker CLI configuration is isolated so user-selected remote contexts,
        # registry credentials, proxies, and arbitrary environment are not used.
        allowed = ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        env["DOCKER_CONFIG"] = str(docker_config)
        if self.docker_host:
            env["DOCKER_HOST"] = self.docker_host
        return env

    def _resolve_binary(self) -> str:
        if not self.docker_executable or "\x00" in self.docker_executable:
            raise SandboxError("Docker executable configuration is invalid.")
        binary = shutil.which(self.docker_executable)
        if binary is None and Path(self.docker_executable).is_file():
            binary = str(Path(self.docker_executable).expanduser().resolve())
        if binary is None:
            raise SandboxError("Docker CLI is unavailable. Install and start a supported local Docker Engine; Ayu will not run tests on the host.")
        self._docker_path = binary
        return binary

    def preflight(self) -> dict[str, Any]:
        """Verify local daemon and exact image availability without pulling or running code."""
        self._validate_image(self.image)
        self._validate_local_host(self.docker_host)
        binary = self._resolve_binary()
        with tempfile.TemporaryDirectory(prefix="ayu-docker-config-") as config_dir:
            config = Path(config_dir)
            try:
                config.chmod(0o700)
            except OSError:
                pass
            env = self._environment(config)
            try:
                info = subprocess.run([binary, "info", "--format", "{{.ServerVersion}}"],
                                      stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=8,
                                      shell=False, env=env)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxError(f"Could not verify the local Docker daemon ({type(exc).__name__}); no host execution fallback is available.") from None
            if info.returncode != 0 or not info.stdout.strip():
                raise SandboxError("Local Docker daemon is unavailable or not ready; Ayu will not execute repository tests on the host.")
            try:
                image = subprocess.run([binary, "image", "inspect", self.image, "--format", "{{json .Config.Env}}|{{.Id}}"],
                                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=8,
                                       shell=False, env=env)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxError(f"Could not inspect the configured local sandbox image ({type(exc).__name__}).") from None
            if image.returncode != 0 or not image.stdout.strip():
                raise SandboxError("The digest-pinned sandbox image is not present locally. Prepare it separately; Ayu never pulls or builds an image during a coding task.")
            if len(image.stdout) > 64 * 1024:
                raise SandboxError("Sandbox image metadata exceeds the 64 KiB inspection bound.")
            raw_env, separator, image_id = image.stdout.strip().partition("|")
            if not separator or not image_id:
                raise SandboxError("Could not inspect sandbox image environment safely.")
            try:
                image_env = json.loads(raw_env)
            except json.JSONDecodeError:
                raise SandboxError("Sandbox image has malformed environment metadata; refusing execution.") from None
            if image_env is None:
                image_env = []
            if (not isinstance(image_env, list) or len(image_env) > 256
                    or any(not isinstance(entry, str) or "=" not in entry for entry in image_env)):
                raise SandboxError("Sandbox image environment metadata exceeds the supported safety policy.")
            keys = []
            for entry in image_env:
                key = entry.split("=", 1)[0]
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key):
                    raise SandboxError("Sandbox image contains an invalid environment variable name.")
                if key not in keys:
                    keys.append(key)
            self.image_env_keys = keys
        return {**SandboxPolicy(self.image).public_dict(), "runtime_version": info.stdout.strip()[:128],
                "image_id": image_id[:128], "image_environment_keys": keys,
                "runtime_endpoint": "local socket syntax (configured socket is operator-trusted)"}

    @staticmethod
    def _checked_paths(workspace: Path, suite: str, paths: list[str]) -> list[str]:
        if suite not in {"pytest", "unittest", "compileall"}:
            raise SandboxError("Supported sandbox suites are pytest, unittest, and compileall.")
        if not isinstance(paths, list) or len(paths) > 5:
            raise SandboxError("At most five relative test paths are allowed.")
        checked: list[str] = []
        root = workspace.resolve(strict=True)
        for raw in paths:
            if (not isinstance(raw, str) or not raw or len(raw) > 512 or "\x00" in raw
                    or "\\" in raw or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive):
                raise SandboxError("Sandbox paths must be workspace-relative POSIX paths.")
            relative = Path(raw)
            if relative.is_absolute() or not relative.parts or any(
                    part in {"", ".", ".."} or part.startswith((".", "-")) for part in relative.parts):
                raise SandboxError("Hidden, absolute, and parent-traversal sandbox paths are rejected.")
            current = root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise SandboxError("Symbolic links are not allowed in sandbox test paths.")
            try:
                resolved = current.resolve(strict=True)
            except OSError:
                raise SandboxError(f"Sandbox test path does not exist: {relative.as_posix()}") from None
            if not resolved.is_relative_to(root) or not (resolved.is_file() or resolved.is_dir()):
                raise SandboxError("Sandbox test targets must be regular files or directories inside the task workspace.")
            if resolved.is_file() and (resolved.name.casefold() in DENIED_NAMES or resolved.suffix.casefold() in DENIED_SUFFIXES):
                raise SandboxError("Secret/database files cannot be test targets.")
            checked.append(relative.as_posix())
        if suite == "unittest":
            if len(checked) > 1:
                raise SandboxError("unittest accepts at most one discovery directory.")
            if checked and not (root / checked[0]).is_dir():
                raise SandboxError("unittest test path must be a discovery directory.")
            if not checked and not (root / "tests").is_dir():
                raise SandboxError("Default unittest discovery requires a tests/ directory.")
        return checked

    @staticmethod
    def _snapshot(source: Path, destination: Path) -> dict[str, Any]:
        """Copy regular repository content without Git metadata, secrets, or links."""
        source = source.resolve(strict=True)
        if not source.is_dir() or source.is_symlink():
            raise SandboxError("Sandbox source workspace must be an existing real directory.")
        destination.mkdir(mode=0o755)
        try:
            destination.chmod(0o755)
        except OSError:
            pass
        copied_files = 0
        copied_bytes = 0
        visited_entries = 0
        omitted_count = 0
        omitted_hash = hashlib.sha256()
        pruned_dirs = PRIVATE_DIRS | {".venv", "venv", "build", "dist"}
        source_device = source.stat().st_dev
        snapshot_deadline = time.monotonic() + SNAPSHOT_TIMEOUT_SECONDS

        mount_points: set[str] = set()
        mount_table = Path("/proc/self/mountinfo")
        if os.name != "nt" and mount_table.exists():
            try:
                with mount_table.open("r", encoding="utf-8", errors="replace") as mount_file:
                    raw_mounts = mount_file.read(1024 * 1024 + 1)
                if len(raw_mounts) > 1024 * 1024:
                    raise SandboxError("Host mount table exceeds the 1 MiB snapshot-safety bound.")
                for line in raw_mounts.splitlines():
                    columns = line.split()
                    if len(columns) < 5:
                        raise SandboxError("Host mount table is malformed; refusing to snapshot nested paths.")
                    decoded = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), columns[4])
                    mount_points.add(os.path.normcase(os.path.abspath(decoded)))
            except OSError:
                raise SandboxError("Could not inspect host mount points before snapshot creation.") from None

        def omit(relative: str) -> None:
            nonlocal omitted_count
            encoded = relative.encode("utf-8", errors="replace")
            omitted_hash.update(len(encoded).to_bytes(4, "big"))
            omitted_hash.update(encoded)
            omitted_count += 1

        pending: list[tuple[Path, Path]] = [(source, Path())]
        while pending:
            if time.monotonic() > snapshot_deadline:
                raise SandboxError(f"Sandbox snapshot exceeded its {SNAPSHOT_TIMEOUT_SECONDS}-second preparation deadline.")
            current, rel_dir = pending.pop()
            try:
                with os.scandir(current) as iterator:
                    entries = []
                    for entry in iterator:
                        if time.monotonic() > snapshot_deadline:
                            raise SandboxError(f"Sandbox snapshot exceeded its {SNAPSHOT_TIMEOUT_SECONDS}-second preparation deadline.")
                        visited_entries += 1
                        if visited_entries > MAX_SNAPSHOT_ENTRIES:
                            raise SandboxError(f"Sandbox workspace exceeds the {MAX_SNAPSHOT_ENTRIES:,}-entry traversal bound.")
                        entries.append(entry)
            except SandboxError:
                raise
            except OSError:
                raise SandboxError("Sandbox snapshot could not enumerate the entire task workspace safely.") from None
            entries.sort(key=lambda item: item.name)
            child_dirs: list[tuple[Path, Path]] = []
            for entry in entries:
                name = entry.name
                src = current / name
                relative_path = rel_dir / name
                relative = relative_path.as_posix()
                folded = name.casefold()
                if folded in pruned_dirs:
                    omit(relative)
                    continue
                if name.startswith("."):
                    safe_hidden = folded in {".gitignore", ".gitattributes", ".env.example"}
                    if not safe_hidden or not entry.is_file(follow_symlinks=False):
                        omit(relative)
                        continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        info = src.lstat()
                        if not (info.st_mode & 0o170000) == 0o040000:
                            raise SandboxError("Sandbox snapshot found an unsupported directory entry.")
                        canonical_dir = os.path.normcase(os.path.abspath(src))
                        if info.st_dev != source_device or os.path.ismount(src) or canonical_dir in mount_points:
                            omit(relative)
                            continue
                        target_dir = destination / relative_path
                        target_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
                        try:
                            target_dir.chmod(0o755)
                        except OSError:
                            pass
                        child_dirs.append((src, relative_path))
                        continue
                    if entry.is_symlink():
                        raise SandboxError("Sandbox snapshot refuses symbolic links; remove links from the task workspace and retry.")
                    if not entry.is_file(follow_symlinks=False):
                        omit(relative)
                        continue
                    if (folded in PRIVATE_FILE_NAMES or (folded.startswith(".env") and folded != ".env.example")
                            or folded in DENIED_NAMES or src.suffix.casefold() in PRIVATE_SUFFIXES):
                        omit(relative)
                        continue
                    before = src.lstat()
                    if not (before.st_mode & 0o170000) == 0o100000:
                        omit(relative)
                        continue
                    if before.st_size > MAX_FILE_BYTES:
                        raise SandboxError(f"Sandbox snapshot file exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MiB per-file bound: {relative}")
                    if before.st_dev != source_device or before.st_nlink > 1:
                        omit(relative)
                        continue
                    copied_files += 1
                    if copied_files > MAX_SNAPSHOT_FILES:
                        raise SandboxError("Sandbox snapshot contains too many files (maximum 10,000).")
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    fd = os.open(src, flags)
                    with os.fdopen(fd, "rb") as inp:
                        actual = os.fstat(inp.fileno())
                        if (not os.path.isfile(src) or actual.st_ino != before.st_ino
                                or actual.st_size != before.st_size or actual.st_nlink > 1):
                            raise SandboxError("Workspace file changed during snapshot; refusing an inconsistent copy.")
                        out_path = destination / relative_path
                        out_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                        bytes_this_file = 0
                        with out_path.open("xb") as out:
                            while True:
                                if time.monotonic() > snapshot_deadline:
                                    raise SandboxError(f"Sandbox snapshot exceeded its {SNAPSHOT_TIMEOUT_SECONDS}-second preparation deadline.")
                                chunk = inp.read(1024 * 1024)
                                if not chunk:
                                    break
                                bytes_this_file += len(chunk)
                                if bytes_this_file > MAX_FILE_BYTES:
                                    raise SandboxError("Workspace file grew beyond the per-file snapshot bound while copying.")
                                if copied_bytes + bytes_this_file > MAX_SNAPSHOT_BYTES:
                                    raise SandboxError("Sandbox snapshot grew beyond the 256 MiB total bound while copying.")
                                out.write(chunk)
                        after = os.fstat(inp.fileno())
                        if (bytes_this_file != before.st_size or after.st_size != before.st_size
                                or after.st_ino != before.st_ino or after.st_nlink > 1
                                or after.st_mtime_ns != before.st_mtime_ns or after.st_ctime_ns != before.st_ctime_ns):
                            raise SandboxError("Workspace file changed during snapshot; refusing an inconsistent copy.")
                        copied_bytes += bytes_this_file
                        executable = 0o111 if (before.st_mode & 0o111) else 0
                        out_path.chmod(0o444 | executable)
                except SandboxError:
                    raise
                except OSError as exc:
                    raise SandboxError(f"Could not copy a regular workspace file into the private sandbox snapshot ({type(exc).__name__}).") from None
            pending.extend(reversed(child_dirs))
        return {"files_copied": copied_files, "bytes_copied": copied_bytes,
                "entries_visited": visited_entries, "paths_omitted": omitted_count,
                "omitted_paths_sha256": omitted_hash.hexdigest()}

    @staticmethod
    def _suite_command(suite: str, paths: list[str]) -> list[str]:
        if suite == "pytest":
            return ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--", *paths]
        if suite == "unittest":
            return ["python", "-m", "unittest", "discover", "-s", paths[0] if paths else "tests", "-v"]
        if suite == "compileall":
            return ["python", "-m", "compileall", "-q", *(paths or ["."])]
        raise SandboxError("Unsupported sandbox test suite.")

    def _container_command(self, workspace: Path, suite_command: list[str], container_name: str,
                           image_env_keys: list[str] | None = None) -> list[str]:
        assert self._docker_path is not None
        mount_source = str(workspace.resolve(strict=True))
        if "," in mount_source:
            raise SandboxError("Sandbox workspace paths containing commas are not supported by Docker mount syntax.")
        tmpfs = f"/tmp:rw,nosuid,nodev,noexec,size={SandboxPolicy(self.image).tmpfs_bytes},mode=1777"
        command = [self._docker_path, "run", "--pull=never", "--rm", "--init", "--name", container_name,
                "--network=none", "--read-only", "--user", "65532:65532",
                "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                "--pids-limit=64", "--memory=536870912", "--memory-swap=536870912",
                "--cpus=1.0", "--ipc=private", "--shm-size=16m", "--tmpfs", tmpfs,
                "--mount", f"type=bind,source={mount_source},target=/workspace,readonly",
                "--workdir=/workspace", "--env=HOME=/tmp", "--env=PYTHONNOUSERSITE=1",
                "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONPYCACHEPREFIX=/tmp/pycache",
                "--env=TMPDIR=/tmp", "--env=TEMP=/tmp", "--env=TMP=/tmp", "--env=PYTHONUNBUFFERED=1",
                "--env=PYTHONPATH=", "--env=PYTHONHOME=", "--env=PYTHONSTARTUP=",
                "--env=PYTHONINSPECT=0", "--env=PYTHONUSERBASE=/tmp/userbase",
                "--env=LD_PRELOAD=", "--env=LD_LIBRARY_PATH=", "--env=DYLD_INSERT_LIBRARIES=",
                "--env=PATH=/usr/local/bin:/usr/bin:/bin", "--entrypoint=python", self.image,
                *suite_command[1:]]
        # Docker images may bake credentials, Python hooks, or proxy settings
        # into Config.Env. Blank every inherited key except names we explicitly
        # set with a sanitized value below. Do not record image values.
        safe_keys = {argument.split("=", 2)[1] for argument in command if argument.startswith("--env=")}
        clear_arguments = [f"--env={key}=" for key in image_env_keys or [] if key not in safe_keys]
        insert_at = command.index("--mount")
        command[insert_at:insert_at] = clear_arguments
        return command

    @staticmethod
    def _drain(pipe: Any, target: bytearray, truncated: list[bool]) -> None:
        while True:
            chunk = pipe.read(8192)
            if not chunk:
                return
            remaining = MAX_CAPTURE_BYTES - len(target)
            if remaining > 0:
                target.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True

    def _remove_container(self, name: str, config: Path) -> bool:
        if self._docker_path is None:
            return False
        try:
            result = subprocess.run([self._docker_path, "rm", "--force", name],
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=8,
                                    shell=False, env=self._environment(config))
            # Not-found means it already exited and --rm removed it.
            return result.returncode == 0 or b"No such container" in result.stderr
        except (OSError, subprocess.TimeoutExpired):
            return False

    def cleanup_container(self, name: str) -> bool:
        """Force-remove one audited Ayu container by exact generated name."""
        if not isinstance(name, str) or not re.fullmatch(r"ayu-test-[a-z0-9-]{1,32}-[a-f0-9]{8}", name):
            raise SandboxError("Refusing cleanup: container name is not a valid Ayu test-run identifier.")
        self._validate_local_host(self.docker_host)
        self._resolve_binary()
        with tempfile.TemporaryDirectory(prefix="ayu-docker-config-") as config_dir:
            config = Path(config_dir)
            try:
                config.chmod(0o700)
            except OSError:
                pass
            env = self._environment(config)
            try:
                info = subprocess.run([self._docker_path, "info", "--format", "{{.ServerVersion}}"],
                                      stdin=subprocess.DEVNULL, capture_output=True, timeout=8,
                                      shell=False, env=env)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxError(f"Could not verify the local Docker daemon for cleanup ({type(exc).__name__}).") from None
            if info.returncode != 0 or not info.stdout.strip():
                raise SandboxError("Local Docker daemon is unavailable; orphan cleanup was not attempted.")
            return self._remove_container(name, config) and self._verify_cleanup(name, config)

    def _verify_cleanup(self, name: str, config: Path) -> bool:
        if self._docker_path is None:
            return False
        try:
            result = subprocess.run([self._docker_path, "inspect", name, "--format", "{{.Id}}"],
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=8,
                                    shell=False, env=self._environment(config))
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode == 0:
            return False
        error = result.stderr.decode("utf-8", errors="replace").casefold()
        return "no such object" in error or "no such container" in error

    def run_tests(self, workspace: Path, suite: str, paths: list[str], timeout: int, *, run_id: str,
                  on_start: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Execute only an allow-listed suite in a disposable, secret-filtered snapshot."""
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 600:
            raise SandboxError("Sandbox test timeout must be between 1 and 600 seconds.")
        source = workspace.expanduser().resolve(strict=True)
        if not source.is_dir():
            raise SandboxError("Sandbox test workspace must be a directory.")
        checked = self._checked_paths(source, suite, paths)
        policy = self.preflight()
        safe_id = re.sub(r"[^a-z0-9-]", "", str(run_id).lower())[:16] or uuid.uuid4().hex[:12]
        container_name = f"ayu-test-{safe_id}-{uuid.uuid4().hex[:8]}"
        cmd = self._suite_command(suite, checked)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="ayu-sandbox-", ignore_cleanup_errors=True) as tmp:
            parent = Path(tmp)
            try:
                parent.chmod(0o700)
            except OSError:
                pass
            snapshot = parent / "workspace"
            snapshot_info = self._snapshot(source, snapshot)
            command = self._container_command(snapshot, cmd, container_name, self.image_env_keys)
            with tempfile.TemporaryDirectory(prefix="ayu-docker-config-") as config_dir:
                config = Path(config_dir)
                try:
                    config.chmod(0o700)
                except OSError:
                    pass
                env = self._environment(config)
                stdout, stderr = bytearray(), bytearray()
                out_truncated, err_truncated = [False], [False]
                if on_start is not None:
                    on_start({"backend": "docker", "container_name": container_name, "image": self.image,
                              "policy_version": policy["policy_version"], "command": cmd,
                              "snapshot": snapshot_info, "timeout_seconds": timeout})
                try:
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, shell=False, env=env,
                                            start_new_session=(os.name != "nt"))
                except OSError as exc:
                    raise SandboxError(f"Could not start Docker execution ({type(exc).__name__}); no host execution fallback is available.") from None
                threads = [threading.Thread(target=self._drain, args=(proc.stdout, stdout, out_truncated), daemon=True),
                           threading.Thread(target=self._drain, args=(proc.stderr, stderr, err_truncated), daemon=True)]
                for thread in threads:
                    thread.start()
                timed_out = False
                cleanup: bool | None = None
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    cleanup = self._remove_container(container_name, config)
                    self._stop_client(proc)
                    cleanup = cleanup and self._verify_cleanup(container_name, config)
                except KeyboardInterrupt:
                    cleanup = self._remove_container(container_name, config)
                    self._stop_client(proc)
                    cleanup = cleanup and self._verify_cleanup(container_name, config)
                    raise
                finally:
                    for thread in threads:
                        thread.join(timeout=5)
                if cleanup is None:
                    cleanup = self._verify_cleanup(container_name, config)
                out = bytes(stdout).decode("utf-8", errors="replace")
                err = bytes(stderr).decode("utf-8", errors="replace")
                truncated = out_truncated[0] or err_truncated[0]
                if truncated:
                    err += "\n[combined container output truncated at 256 KiB per stream]"
                infrastructure = proc.returncode is None or (timed_out and cleanup is False)
                if proc.returncode in {125, 126, 127} and not timed_out:
                    # Docker reserves 125/126/127 for runtime/command setup errors;
                    # a Python test's nonzero exit remains an ordinary test result.
                    infrastructure = True
                result = {
                    "command": cmd,
                    "exit_code": proc.returncode,
                    "timed_out": timed_out,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "stdout": out,
                    "stderr": err,
                    "infrastructure_error": infrastructure,
                    "output_truncated": truncated,
                    "backend": "docker",
                    "image": self.image,
                    "policy_version": policy["policy_version"],
                    "container_name": container_name,
                    "cleanup_confirmed": cleanup,
                    "snapshot": snapshot_info,
                    "policy": policy,
                }
                return result

    @staticmethod
    def _stop_client(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def status(self) -> dict[str, Any]:
        """Return policy/runtime readiness; failures are explicit and do not execute code."""
        try:
            return {"ready": True, **self.preflight()}
        except SandboxError as exc:
            return {"ready": False, "backend": "docker", "policy_version": "ayu-docker-v1", "reason": str(exc)}
