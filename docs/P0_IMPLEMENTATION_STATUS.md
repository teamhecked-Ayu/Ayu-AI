# P0 implementation status

**Branch:** `feat/p0-execution-security`
**Base:** `feature/local-runtime-contract`
**Main:** unchanged.

## Implemented in this branch

- Immutable, versioned execution contracts for local agents, explicitly approved remote-context agents, and local coding runs.
- Deny-by-default capability authorization. Unknown capabilities are not defined; local agents cannot write or execute processes; coding writes are limited to a task worktree; process execution is allowed only through the fixed Docker sandbox policy.
- Workspace identity checks prevent tools and edits from crossing the contract's selected workspace.
- Application-level budgets for model calls, tool calls, sandbox invocations, cumulative model-output bytes, and elapsed wall time.
- Tool registry permission metadata is enforced when a registry is bound to a contract.
- Agent/coding runs persist their execution contract as audit evidence.
- Agent audit events use a SHA-256 previous-event hash chain. Event rows and integrity rows are written in one SQLite transaction. Legacy event records are backfilled without changing their payloads.
- `ayu audit verify RUN_ID` checks event sequence continuity, previous-hash links, and payload/event hashes. Agent and coding inspect output include integrity status.
- Focused unit tests cover capability denial, remote consent, workspace scope, immutable contracts, resource budgets, audit tampering, and sequence gaps.
- The opt-in real-Docker suite now checks non-root identity, dropped capabilities, no-new-privileges, read-only root/workspace mounts, no-exec writable temporary storage, network isolation, cgroup memory/CPU/PID limits, and verified cleanup after timeout.
- A manual GitHub Actions path runs those real-Docker checks with a digest-pinned Docker Official Python image. The regular cross-platform test matrix remains offline and does not require Docker.

## Verification still required

- The manual real-Docker workflow has been added but has **not yet been run**. A passing mocked unit test is not evidence of actual container enforcement.
- The real-Docker workflow covers a GitHub-hosted Linux runner. Test the supported local Docker configurations on Linux, Windows, and macOS before claiming cross-platform runtime isolation.
- Review runtime behavior under daemon failure, forced cancellation, and host shutdown. The timeout test verifies the normal forced-removal path; it cannot prove cleanup after power loss or daemon failure.
- Termux/Android remains untested and is not claimed as supported for Docker-backed coding.
- Application wall-clock budgets are cooperative checks. They do not forcibly interrupt arbitrary blocked Python operations; each runtime/tool transport must enforce its own timeout.
- The local SQLite event hash chain is tamper-evident, not externally signed. A database administrator can rewrite the entire chain or truncate the database; independent retention requires external log export or signatures anchored outside the database.

## Safety boundary

P0 does not grant arbitrary shell execution, agent network access, remote coding, or production autonomy. Docker configuration is an application policy, not a proof of container-escape resistance. Unit tests, CI, and model review do not establish a perfect OS security boundary. No changes are made to `main`, no existing branch is force-updated, and no PR is merged by this work.
