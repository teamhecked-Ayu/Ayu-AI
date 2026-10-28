# Ayu AI engineering status — durable coding-run recovery

**Date:** 2026-10-28
**Repository:** `teamhecked-Ayu/Ayu-AI`
**Baseline:** `feature/local-runtime-contract` at `ba43c0807497a90cf0cefa6d7a812b432b3963ed`; clean and pushed, eight commits ahead of `main` (`4f72c099`). No pull request exists. The recovery implementation was committed and pushed as `b33b88f088c42e160d71ad9c5e4d51264aa45bc8`; `main` remains unchanged and there is no pull request.

## Project map (source-code inspection)

| Capability | Main implementation | Existing verification | Status and key gap |
|---|---|---|---|
| CLI/configuration | `ayu/cli.py`, `ayu/settings.py` | CLI and settings unit tests; cross-platform Actions on prior sandbox commit | **UNIT-TESTED.** Basic CLI stays usable without a model; runtime-specific commands require their prerequisites. |
| Model providers | `ayu/provider.py` | Mocked provider tests; earlier local llama.cpp loopback inference recorded in delivery report | **PARTIAL.** Local llama-server inference had one runtime check; live remote calls and broad model compatibility are unverified. |
| Persistence/memory/projects/tasks | `ayu/store.py`, `ayu/memory.py`, `ayu/project.py` | SQLite temporary-database tests, migration and backup/restore tests | **UNIT-TESTED.** Single-user local SQLite design; no multi-user guarantees. |
| Read-only agent | `ayu/agents.py`, `ayu/tools.py` | Deterministic structured-response mocks and policy tests | **UNIT-TESTED, NOT REAL-MODEL-EVALUATED.** Sequential bounded tools, no shell; model review is not independent proof. |
| Coding agent | `ayu/coding.py`, `ayu/cli.py` | Patch/worktree/workflow tests with fake provider and fake sandbox | **UNIT-TESTED.** Local llama-server JSON-schema path, scoped edits and bounded correction are implemented. Current end-to-end path has not completed a task with the required real sandbox. |
| Docker test boundary | `ayu/sandbox.py`, `tests/test_sandbox_integration.py` | Policy/snapshot tests with mocks; opt-in real Docker integration | **POLICY IMPLEMENTED; REAL ISOLATION BLOCKED/UNVERIFIED.** No Docker/Podman/bubblewrap/socket or network namespace is available here. No host fallback exists. |
| Coding run/audit lifecycle | `ayu/store.py`, `ayu/coding.py` | SQLite run/event and new recovery fault-injection tests | **RECOVERY IMPROVED, PARTIAL.** Durable run identity and explicit `unknown_outcome` transition now exist. No automatic crash detection, resource reconciliation, or resume. |
| AyuLM | `ayu/ayulm.py`, CLI training commands | CPU synthetic smoke and CI CPU job | **SOFTWARE-PATH VERIFIED, MODEL QUALITY UNVERIFIED.** Smoke is one training step on tiny synthetic data, not a quality benchmark. |
| Local document search | `ayu/tools.py`, `ayu/cli.py` | Bounded lexical-search tests | **IMPLEMENTED/UNIT-TESTED.** No embedding/vector index or semantic retrieval. |

## Selected milestone

**Objective:** Make interrupted coding runs explicitly reconcilable without falsely claiming success or repeating side effects. The immediate Docker boundary work remains blocked because this environment has no usable runtime, and arbitrary images were not downloaded or trusted. Recovery is an independent, safe next milestone already supported by existing `agent_runs` and `agent_events` records.

**Scope:** add a backward-compatible SQLite status migration; atomically persist a coding run identity event with its `running` row; provide a manual operator-confirmed transition to `unknown_outcome`; expose `ayu code recover`; add fault-injection and legacy-migration tests; update operational/security documentation.

**Non-goals:** automatic detection of process death, deciding whether tests completed, rerunning a task, deleting worktrees, stopping/removing containers during state reconciliation, sandbox verification, or changing `main`.

**Acceptance criteria and evidence:**

- A run and its sequence-zero coding identity event commit in the same SQLite transaction; duplicate run creation does not append a stray event.
- An operator can transition only a still-running record with coding evidence after explicitly confirming the Ayu process stopped. The state and recovery event commit atomically; repeating the operation is idempotent.
- The transition refuses missing confirmation, missing/non-coding evidence, and terminal runs. It never retries the task or performs cleanup.
- Existing database rows/events survive the additive table-constraint migration; `PRAGMA foreign_key_check` is clean; reopening the database is idempotent.
- A fault injected at temporary-run-directory creation is recorded against the already-durable run and ends in `failed`, rather than leaving an unidentifiable `running` record.
- A disposable child process is force-terminated after persisting a run; a fresh store instance reopens the database and marks the surviving running record `unknown_outcome` without retrying it.
- **GitHub CI:** Actions run [40661096752](https://github.com/teamhecked-Ayu/Ayu-AI/actions/runs/40661096752) passed on recovery commit `b33b88f`: Ubuntu/macOS/Windows across Python 3.10/3.11/3.12 (nine core combinations), plus the CPU AyuLM train/checkpoint/generation job. Gitleaks [run 40661096704](https://github.com/teamhecked-Ayu/Ayu-AI/actions/runs/40661096704) also passed. This verifies CI tests and scanning, not hard-crash recovery or Docker isolation.
- **Local verification:** `python -m pytest -q -rs` reported **118 passed, 2 skipped, 16 subtests passed**. Skips: opt-in real llama-server test requires explicit endpoint/model; real Docker integration requires `AYU_TEST_SANDBOX_IMAGE` and Docker. `python -m compileall -q ayu tests`, one-step `python -m ayu.cli test --ayulm-smoke`, Python 3.10/3.11 grammar parsing for 31 source/test files, and `git diff --check` passed. The AyuLM smoke is software-path evidence, not model-quality evidence.
- **Still unverified:** real Docker security integration remains blocked; host shutdown/power-loss durability, concurrent Ayu process coordination, container/worktree reconciliation and the full crash matrix were not tested. The child-process termination test is a bounded disposable SQLite scenario, not evidence of power-loss durability.

## Next priorities

1. Review and run actual Docker boundary integration using an organization-reviewed, locally present digest-pinned image on supported native and Desktop/rootless runtimes; do not infer isolation from CI unit tests.
2. Extend recovery with safe resource reconciliation and crash injection at worktree creation, container startup, event persistence, cleanup, and final publication; keep uncertain outcomes explicit and avoid auto-retry.
3. Build a deterministic coding-evaluation harness with reviewed fixture tasks and independent oracles before making claims about model capability.

**Environment limitation:** this sandbox has no `docker`, `podman`, `bubblewrap`/`bwrap`, or Docker socket. `unshare -n true` failed with `Operation not permitted`. Real filesystem, network, resource, cancellation, and orphan-cleanup behavior are therefore **BLOCKED / NOT VERIFIED** here.
