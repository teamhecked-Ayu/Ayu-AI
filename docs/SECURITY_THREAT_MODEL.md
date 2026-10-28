# Ayu AI coding-agent threat model

**Status:** The application-level Docker execution policy is implemented and unit-tested with mocks. **Real-runtime security is BLOCKED / NOT VERIFIED**: the current development environment has no Docker/Podman/bubblewrap executable or Docker socket, and `unshare -n true` fails with `Operation not permitted`. No sandbox test has run in a real container here. See README's platform/test section for the verification context.

## Scope and security objective

This covers local model-assisted coding from repository inspection through patching, test execution, audit persistence, and cleanup. The objective is to prevent untrusted repository content and model-generated code from reading or changing host data, exposing credentials, making unauthorized network requests, exhausting the host, or damaging unrelated work. Ayu cannot prove that a model's reasoning is safe. Security must be enforced by trusted code and the operating system/runtime, not by prompt instructions or model review.

## Assets and trust boundaries

Assets include the user's home directory and credentials, unrelated repositories and files, the source repository's Git metadata and original working tree, local model/provider configuration, task data and audit records, and host availability. The trusted computing base includes Ayu's CLI/runner, the local Docker CLI/daemon, the host kernel, and the selected digest-pinned sandbox image. An immutable digest prevents tag drift; it does not establish image provenance, safety, or vulnerability status.

Untrusted inputs include the user task, repository names and paths, README/docstring/comment/test content, dependencies already present in the repository, test output, and every model response. Prompt-injection text is data, not authority. Model-generated patches and tests are never a source of execution policy.

## Implemented controls (application behavior; not yet real-runtime verified)

- `ayu code run` requires local inference through the strict-JSON llama-server backend and checks Docker/image readiness before provider construction or model calls. It has no host-test fallback and no user-selectable shell command.
- The Docker client environment is reduced and its configuration is isolated in a private temporary directory. Remote TCP/context use is refused. Explicit POSIX endpoints must be existing Unix sockets, but Ayu cannot prove that a user-managed socket proxy does not forward remotely; configured socket endpoints remain operator-trusted.
- A digest-pinned image must already exist locally; the runner never pulls or builds images. A fresh disposable snapshot is made per test run; `.git` (directory or worktree pointer file), common credential directories/names/extensions, and most hidden files/config are omitted. Symlinks, hard-linked files, nested mount points, and cross-device entries are refused or omitted. Directory/file traversal is capped at 20,000 entries, copied files at 10,000, each file at 32 MiB, copied content at 256 MiB, and preparation time at a cooperative 60-second deadline between filesystem operations; a blocked kernel/filesystem call can exceed it. Byte limits are enforced while reading and file metadata rechecked after copy. Omitted paths are counted/hashed incrementally rather than accumulated in memory. Files are copied as read-only and the snapshot parent is private.
- Test execution uses a fixed Python command for `pytest`, `unittest`, or `compileall`. The container command requests `--network=none`, `--read-only`, UID/GID `65532:65532`, `--cap-drop=ALL`, `no-new-privileges`, PID limit 64, 512 MiB memory and swap, one CPU, private IPC, 16 MiB shared memory, and a 64 MiB `/tmp` tmpfs. The task snapshot is mounted read-only. Output is bounded to 256 KiB per stream and wall-clock duration to at most 600 seconds.
- The run identity/container name is durably recorded before Docker `run` is launched. `--rm` is requested; normal completion inspects for container removal. Timeout/cancellation attempts forced removal. `ayu sandbox cleanup RUN_ID` can force-remove only generated Ayu container names recorded in that run's audit events. After verifying that no Ayu process is still executing a run, an operator can use `ayu code recover RUN_ID --confirm-process-stopped` to atomically classify a persisted running coding run as `unknown_outcome`; it never retries work or removes resources.
- Audit output is hashed and best-effort redacted; correction/review prompts receive bounded sanitized output. Redaction recognizes common token/assignment/key patterns but is not exhaustive.

## Threats and residual risks

| Threat | Current mitigation | Residual risk / verification |
|---|---|---|
| Malicious/compromised repo or prompt injection | Untrusted content; fixed commands; schema/path validation; no model shell | Tests are arbitrary code. Prompting is not a security boundary. |
| Host filesystem access | No home/source Git/secret mount; bounded disposable snapshot; read-only bind/root; symlink, hardlink and nested-mount protections | Container daemon/kernel remain trusted. Hostile image or container escape can defeat controls. Secret filtering is filename-based, not content scanning; unusually named secrets inside the task repository may be copied. Explicit Unix socket proxies are operator-trusted and may forward remotely. |
| Network exfiltration | `--network=none`; remote TCP/context endpoints rejected; Unix sockets type-checked | Not tested on real Docker here; a configured local socket can proxy a remote daemon; runtime/daemon bugs or covert channels remain possible. |
| Dependency/build hooks | No auto-install/build; image must be locally prepared and digest pinned | Image provenance, installed packages, Dockerfile, and image vulnerability review are operator responsibilities. |
| Resource exhaustion | Memory/CPU/PID/tmpfs/wall-time/output limits requested | Actual cgroup enforcement and disk/daemon pressure are unverified. Some Docker Desktop/rootless configurations may reject or reinterpret limits; runner must fail on runtime errors. |
| Child processes/timeout | PID namespace and `--init`; timeout requests `docker rm --force` and stops CLI; cleanup verification | Not verified under real workload. `SIGKILL`, machine shutdown, or daemon failure may leave an orphan. If the host process crashes after audit intent but before cleanup, operator can run `ayu sandbox cleanup RUN_ID`; cleanup itself needs a functioning local daemon. |
| Container writes to task source | Snapshot is read-only in code and Docker bind requested read-only | Mock construction confirms flags only, not daemon enforcement. |
| Secret exposure in prompt/audit | No host env forwarded to container/model; bounded sanitization; local model only | Arbitrary secrets embedded in source/test output cannot be guaranteed redacted. Do not store credentials in coding repositories. |
| Git hooks/filters and original source | Local Git checks, hooks path disabled, source untouched; task branch/worktree for edits | The worktree is not a security boundary; local Git and host Ayu run with user privileges. |
| Crash and continuity | Startup container intent persisted; run IDs and audit events; operator-confirmed transition to `unknown_outcome`; separate explicit container cleanup | Recovery depends on the operator truthfully confirming the process stopped. It does not establish whether tests/review completed, does not clean the worktree/container, and does not resume or retry. Automatic crash detection/reconciliation and proving completion after a hard crash are not implemented. |

## Control categories (do not conflate them)

- **Application-level validation:** schema/path/patch checks, fixed suite selection, output bounds, Git cleanliness checks. Unit tests can verify these.
- **Tool permissions:** read-only tools are bounded; coding adds controlled edits. A tool allow-list is not an OS sandbox.
- **OS/container process isolation:** requested via Docker namespaces, non-root UID, dropped capabilities, and PID/cgroup limits. **REAL-RUNTIME-VERIFIED: NO.**
- **Filesystem isolation:** requested through a private disposable copy, read-only bind mount, and no home/source `.git` mount. **REAL-RUNTIME-VERIFIED: NO.**
- **Network isolation:** requested with `--network=none`. **REAL-RUNTIME-VERIFIED: NO.**
- **Resource controls:** requested through Docker limits. **REAL-RUNTIME-VERIFIED: NO.**
- **Human approval:** not required as a substitute for sandboxing. Docker/image readiness is mandatory; no host-execution consent switch exists.
- **Audit logging:** records policy, generated container name, command class, output hashes/excerpts, return code, cleanup status, and the operator's unknown-outcome recovery transition. Logs are observable evidence, not proof of enforced policy or of a truthful operator attestation.

## Docker assumptions and image preparation

The initial backend is local Docker Engine/Desktop only. Set `AYU_SANDBOX_IMAGE` to a locally present immutable `repository@sha256:<64 lowercase hex>` image that has Python and the selected test tools installed. Ayu never pulls/builds or installs packages during a task. `AYU_DOCKER_BINARY` and `AYU_DOCKER_HOST` can select a CLI/Unix socket; TCP endpoints are refused and the socket must exist, but its proxy/locality characteristics are operator-trusted. `ayu sandbox status` performs daemon/image preflight without starting a container; `ayu sandbox cleanup RUN_ID` acts only on names in that persisted run record.

Docker is not automatically secure. Rootful Docker gives its daemon substantial host authority. The daemon, host kernel, Docker Desktop VM (where applicable), and image remain part of the trusted computing base. `--network=none` and cgroup flags must be validated on each supported runtime/platform. Docker Desktop filesystem/network semantics and rootless user namespaces differ from native rootful Linux. No cross-platform sandbox guarantee is made until real-runtime security tests pass on that OS.

## Current environment / verification boundary

Phase 0 inspection found no Docker, Podman, bubblewrap, or Docker socket; only `unshare` was installed and the network namespace operation failed with `Operation not permitted`. The sandbox runner's unit tests use fake subprocesses and inspect command policy; they do **not** run untrusted code in a container. Therefore actual filesystem, network, resource, secret-isolation, cancellation, and orphan-cleanup tests are blocked here. The autonomous CLI deliberately refuses to run tests without a locally ready digest-pinned Docker image and daemon. This preserves fail-closed behavior but means full coding-run runtime verification cannot be claimed from this environment.

## Verification labels

Use these labels in documentation and reports: **IMPLEMENTED**, **UNIT-TESTED**, **INTEGRATION-TESTED**, **REAL-RUNTIME-VERIFIED**, **PLATFORM-VERIFIED**, **NOT IMPLEMENTED**, **NOT VERIFIED**, **BLOCKED**. A mocked runner cannot establish filesystem, network, secret, resource, or process isolation. A passing coding test does not establish model intelligence or security.
