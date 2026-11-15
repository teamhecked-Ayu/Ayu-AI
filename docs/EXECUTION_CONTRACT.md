# Ayu AI execution contract — P0

**Status:** Application-level contract and authorization are implemented and unit-tested. OS isolation remains a separate runtime property and is not implied by this document.

## Contract

Every model-mediated agent run is bound to an immutable `ExecutionContract`. The contract records:

- execution mode (`local` or explicitly approved `remote`)
- one canonical workspace root and scope kind
- a fixed set of capabilities
- a bounded resource budget
- remote-context approval state
- sandbox backend and policy version where applicable

Model output, repository text, tool output, and test output cannot grant additional permissions. Unknown capabilities are denied.

## Capability profiles

| Profile | Capabilities |
|---|---|
| Local read-only agent | `workspace.read`, `model.inference.local` |
| Remote read-only agent | `workspace.read`, `model.inference.remote`, `remote_context.send` (requires explicit CLI approval) |
| Coding run | `workspace.read`, `workspace.write.task_worktree`, `model.inference.local`, `process.execute.sandbox` |

Workspace writes are restricted to a separate task worktree. Process execution is restricted to the fixed test-suite runner through the supported sandbox policy. No shell capability is defined.

## Default budgets

| Run type | Model calls | Tool calls | Sandbox invocations | Total application wall budget |
|---|---:|---:|---:|---:|
| Read-only agent | 3 | up to 4 | 0 | 180 seconds |
| Coding workflow | 5 | 0 | up to 2 | 1,800 seconds |

The coding test runner has its own tighter per-test timeout (1–600 seconds), output cap, memory/CPU/PID limits, and disposable read-only snapshot. Contract time budgets are cooperative application checks; they cannot interrupt a blocked kernel or filesystem operation. OS-level resource limits must be enforced and verified by the sandbox runtime.

## Audit integrity

Each agent event is recorded with a SHA-256 hash chained to the preceding event. The chain covers run ID, sequence, actor, event type, canonical payload text, timestamp, and previous hash. Event creation and integrity metadata are committed in the same SQLite transaction.

Use:

```sh
ayu audit verify RUN_ID
```

The command checks contiguous event numbering, presence of integrity metadata, previous-hash linkage, and event content hashes. Legacy event rows are backfilled during database initialization without changing their payloads.

This is tamper-evident application logging, not an externally anchored signature system. A user with write access to the database can rewrite both event rows and hashes or truncate the database. Use OS file permissions, backups, and external log export if independent retention is required.

## Verification labels

- **IMPLEMENTED / UNIT-TESTED:** contract construction, capability denial, workspace binding, resource-budget checks, and audit-chain verification.
- **NOT REAL-RUNTIME-VERIFIED:** Docker namespace/network/filesystem/resource enforcement, container escape resistance, and crash/power-loss durability.
- **NOT PLATFORM-VERIFIED:** Android/Termux and non-Linux Docker semantics.

No passing unit test or model reviewer output is evidence of an enforced OS security boundary.
