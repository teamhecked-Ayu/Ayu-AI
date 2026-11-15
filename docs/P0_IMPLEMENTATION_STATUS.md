# P0 implementation status

**Branch:** `feat/p0-execution-security`
**Base:** `feature/local-runtime-contract`
**Main:** unchanged.

## Implemented in this branch

- Immutable, versioned execution contracts for local agents, explicitly approved remote-context agents, and local coding runs.
- Deny-by-default capability authorization. Unknown capabilities are not defined; local agents cannot write or execute processes; coding writes are limited to a task worktree; process execution is allowed only through the fixed Docker sandbox policy.
- Workspace identity checks prevent tools and edits from crossing the contract's selected workspace.
- Application-level budgets for model calls, tool calls, sandbox invocations, cumulative model-output bytes, and elapsed wall time.
- Tool registry permission metadata is now enforced when a registry is bound to a contract.
- Agent/coding runs persist their execution contract as audit evidence.
- Agent audit events use a SHA-256 previous-event hash chain. The event row and integrity row are written in one SQLite transaction. Legacy event records are backfilled without changing their contents.
- `ayu audit verify RUN_ID` checks event sequence continuity, previous-hash links, and payload/event hashes. Agent and coding inspect output include integrity status.
- Added focused tests for capability denial, remote consent, workspace scope, immutable contracts, resource budgets, audit tampering, and sequence gaps.

## Verification still required

- Run the full Python suite and cross-platform CI on this branch.
- Execute the opt-in Docker integration suite against an approved, locally available digest-pinned image on a real supported Docker runtime.
- Verify actual filesystem/network/resource isolation, cancellation, orphan cleanup, and runtime behavior on each supported OS.
- Termux/Android remains untested.
- A local SQLite hash chain is tamper-evident, not externally signed. A database administrator can rewrite the entire chain or truncate the database; independent retention requires external log export or signatures anchored outside the database.

## Safety boundary

P0 does not grant arbitrary shell execution, network access to agents, remote coding, or production autonomy. Unit tests and model review do not establish OS isolation. No changes are made to `main`, no existing branch is force-updated, and no PR is merged by this work.
