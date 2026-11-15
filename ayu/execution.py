"""Immutable execution contracts and deny-by-default capability authorization.

Contracts are application policy, not an OS sandbox. Process isolation still depends
on the configured runtime and must be verified independently on each platform.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import time


CONTRACT_VERSION = "ayu-exec-v1"
GENERIC_CAPABILITIES = {
    "workspace.read",
    "workspace.write.task_worktree",
    "model.inference.local",
    "model.inference.remote",
    "remote_context.send",
    "process.execute.sandbox",
}


class ExecutionPolicyError(PermissionError):
    """An operation is outside the immutable execution contract."""


@dataclass(frozen=True)
class ResourceBudget:
    """Per-run ceilings. These are app-level ceilings, not OS resource controls."""

    max_model_calls: int
    max_tool_calls: int
    max_process_invocations: int
    max_wall_seconds: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        bounds = {
            "max_model_calls": (0, 8),
            "max_tool_calls": (0, 4),
            "max_process_invocations": (0, 2),
            "max_wall_seconds": (1, 3600),
            "max_output_bytes": (1024, 1024 * 1024),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in {minimum}..{maximum}.")


@dataclass(frozen=True)
class ExecutionContract:
    """A fixed run scope; a model response cannot add capabilities to it."""

    mode: str
    workspace_root: str
    scope_kind: str
    capabilities: tuple[str, ...]
    budget: ResourceBudget
    remote_context_approved: bool = False
    sandbox_backend: str | None = None
    sandbox_policy_version: str | None = None
    version: str = CONTRACT_VERSION

    @staticmethod
    def _root(path: Path) -> str:
        if not isinstance(path, Path):
            path = Path(path)
        return str(path.expanduser().resolve(strict=False))

    @classmethod
    def for_agent(
        cls,
        workspace: Path,
        *,
        mode: str = "local",
        remote_context_approved: bool = False,
        max_tool_calls: int = 4,
    ) -> "ExecutionContract":
        if mode == "local":
            if remote_context_approved:
                raise ValueError("Remote-context approval is not applicable to local inference.")
            capabilities = ("workspace.read", "model.inference.local")
        elif mode == "remote":
            if remote_context_approved is not True:
                raise ExecutionPolicyError(
                    "Remote agent execution requires explicit approval to send task and workspace context."
                )
            capabilities = ("workspace.read", "model.inference.remote", "remote_context.send")
        else:
            raise ValueError("Agent execution mode must be local or remote.")
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or not 1 <= max_tool_calls <= 4:
            raise ValueError("max_tool_calls must be between 1 and 4.")
        return cls(
            mode=mode,
            workspace_root=cls._root(workspace),
            scope_kind="workspace",
            capabilities=capabilities,
            budget=ResourceBudget(
                max_model_calls=3,
                max_tool_calls=max_tool_calls,
                max_process_invocations=0,
                max_wall_seconds=180,
                max_output_bytes=64 * 1024,
            ),
            remote_context_approved=remote_context_approved,
        )

    @classmethod
    def for_coding(cls, worktree: Path, sandbox_policy: dict[str, Any]) -> "ExecutionContract":
        if not isinstance(sandbox_policy, dict):
            raise ExecutionPolicyError("Coding requires a verified sandbox policy description.")
        backend = sandbox_policy.get("backend")
        policy_version = sandbox_policy.get("policy_version")
        # fake-test-only is a deliberately explicit unit-test adapter. It is not
        # accepted by the CLI's production Docker runner.
        if backend == "docker" and policy_version == "ayu-docker-v1":
            pass
        elif backend == "fake-test-only" and policy_version == "unit-test":
            pass
        else:
            raise ExecutionPolicyError("Coding execution requires the supported Docker policy; no host fallback is allowed.")
        return cls(
            mode="local",
            workspace_root=cls._root(worktree),
            scope_kind="task_worktree",
            capabilities=(
                "workspace.read",
                "workspace.write.task_worktree",
                "model.inference.local",
                "process.execute.sandbox",
            ),
            budget=ResourceBudget(
                max_model_calls=5,
                max_tool_calls=0,
                max_process_invocations=2,
                max_wall_seconds=1800,
                max_output_bytes=256 * 1024,
            ),
            sandbox_backend=backend,
            sandbox_policy_version=policy_version,
        )

    def public_dict(self) -> dict[str, Any]:
        """Safe, serializable policy summary for local audit events."""
        return {
            "version": self.version,
            "mode": self.mode,
            "workspace_root": self.workspace_root,
            "scope_kind": self.scope_kind,
            "capabilities": list(self.capabilities),
            "budget": asdict(self.budget),
            "remote_context_approved": self.remote_context_approved,
            "sandbox_backend": self.sandbox_backend,
            "sandbox_policy_version": self.sandbox_policy_version,
        }


class ExecutionPolicy:
    """Deterministic default-deny authorization for built-in Ayu operations."""

    def authorize(
        self,
        contract: ExecutionContract,
        required: Iterable[str],
        *,
        workspace: Path | None = None,
    ) -> None:
        if not isinstance(contract, ExecutionContract) or contract.version != CONTRACT_VERSION:
            raise ExecutionPolicyError("Missing or unsupported execution contract.")
        requested = tuple(required)
        if not requested or any(not isinstance(item, str) for item in requested):
            raise ExecutionPolicyError("An explicit capability set is required.")
        unknown = set(requested) - GENERIC_CAPABILITIES
        if unknown:
            raise ExecutionPolicyError("Unknown or permanently denied capability requested.")
        missing = set(requested) - set(contract.capabilities)
        if missing:
            raise ExecutionPolicyError("Execution denied; contract does not grant: " + ", ".join(sorted(missing)))
        if "remote_context.send" in requested and contract.remote_context_approved is not True:
            raise ExecutionPolicyError("Remote context transmission has no explicit operator approval.")
        if "workspace.write.task_worktree" in requested and contract.scope_kind != "task_worktree":
            raise ExecutionPolicyError("Workspace writes are restricted to an isolated task worktree.")
        if "process.execute.sandbox" in requested:
            if contract.scope_kind != "task_worktree" or contract.sandbox_backend != "docker":
                if not (contract.sandbox_backend == "fake-test-only" and contract.sandbox_policy_version == "unit-test"):
                    raise ExecutionPolicyError("Process execution is allowed only through the configured Docker sandbox.")
            if contract.sandbox_policy_version not in {"ayu-docker-v1", "unit-test"}:
                raise ExecutionPolicyError("Unsupported sandbox policy version.")
        if workspace is not None and self._root(workspace) != contract.workspace_root:
            raise ExecutionPolicyError("Operation is outside the workspace bound to this execution contract.")

    @staticmethod
    def _root(path: Path) -> str:
        return str(Path(path).expanduser().resolve(strict=False))

    @staticmethod
    def check_budget(
        contract: ExecutionContract,
        *,
        elapsed_seconds: float,
        model_calls: int = 0,
        tool_calls: int = 0,
        process_invocations: int = 0,
        output_bytes: int = 0,
    ) -> None:
        budget = contract.budget
        if elapsed_seconds > budget.max_wall_seconds:
            raise ExecutionPolicyError("Execution exceeded its total wall-clock budget; late results are discarded.")
        if model_calls > budget.max_model_calls:
            raise ExecutionPolicyError("Execution exceeded its model-call budget.")
        if tool_calls > budget.max_tool_calls:
            raise ExecutionPolicyError("Execution exceeded its tool-call budget.")
        if process_invocations > budget.max_process_invocations:
            raise ExecutionPolicyError("Execution exceeded its sandbox-invocation budget.")
        if output_bytes > budget.max_output_bytes:
            raise ExecutionPolicyError("Execution exceeded its cumulative model-output budget.")


def elapsed_since(started_at: float) -> float:
    """Monotonic elapsed duration for cooperative budget enforcement."""
    return max(0.0, time.monotonic() - started_at)
