# Ayu AI / AyuLM

Ayu AI is a terminal-first, local-first software foundation. It is useful without API credentials and never silently falls back to a remote model. AyuLM is a small byte-level decoder Transformer for educational/research training—not a pretrained assistant or a claim of frontier-level capability. The software does not claim autonomous production-OS guarantees.

## Operating modes and optional services

- **Basic:** local CLI, SQLite conversation/memory/project/task records, configuration checks, and bounded lexical text search. It does not pretend to provide LLM reasoning.
- **Local inference:** an explicitly selected, already-installed llama.cpp runtime and a user-supplied compatible GGUF. The preferred `llama-server` backend preserves chat roles for the model's template and checks the server's actual token/context budget; the `llama-cli` adapter remains for backward compatibility. Ayu does not download weights or contact an AI provider in local mode.
- **Remote inference:** explicitly selected OpenAI, Anthropic, or OpenAI-compatible adapter. Remote calls require credentials/model configuration and an explicit remote chat/agent command (or `AYU_MODE=remote`); there is no fallback between providers.
- **NewsAPI:** optional article-metadata search, independent from AI inference. It is disabled unless `AYU_NEWSAPI_ENABLED=true`, a NewsAPI key is configured, and the command includes `--allow-network`. Results are untrusted external metadata and are not automatically sent to an AI provider.
- **AyuLM research:** native byte-level training, held-out validation, checkpoint/resume, evaluation, generation, and local throughput benchmark. PyTorch is optional. A model trained from scratch on a tiny corpus is not a conversational assistant.

## Install

Requires Python 3.10 or later. The basic package has no NumPy/PyTorch requirement (Python 3.10 uses the small `tomli` TOML backport); training, legacy prototype, and developer dependencies are optional extras.

### Linux / macOS

```sh
git clone https://github.com/teamhecked-Ayu/Ayu-AI.git
cd Ayu-AI
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ayu --help
```

### Windows PowerShell

```powershell
git clone https://github.com/teamhecked-Ayu/Ayu-AI.git
Set-Location Ayu-AI
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ayu --help
```

### Android Termux (documented profile; not manually tested)

The intended baseline is the Python/SQLite CLI; native inference/training availability depends on device architecture, RAM, Android/Termux packages, and compatible upstream binaries/wheels. These commands have **not** been verified on an Android device:

```sh
pkg update
pkg install python git
git clone https://github.com/teamhecked-Ayu/Ayu-AI.git
cd Ayu-AI
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ayu doctor
ayu test
```

If Termux's Python packaging policy or a wheel/native dependency prevents installation, do not bypass that policy blindly; retain the basic CLI profile or use the appropriate Termux package. GGUF files can be large; check available storage/RAM and the model license first. No Docker, GPU, or model download is required for basic commands.

### Optional CPU AyuLM training dependency

Install only if you want native AyuLM training/evaluation/generation and a compatible PyTorch wheel is available:

```sh
python -m pip install -e '.[training]'
ayu test --ayulm-smoke
```

The smoke test uses a generated synthetic corpus and temporary checkpoint; it does not touch your documents or leave model artifacts behind. CUDA is optional and not required for CPU use.

The retained legacy NumPy prototype is separate from the native byte-level AyuLM model; install it only when importing the old compatibility classes with `python -m pip install -e '.[prototype]'`.

## First run and local configuration

```sh
ayu --help
ayu --version
ayu init                    # create .env; does not overwrite an existing file by default
ayu doctor                  # local checks; sends no network request
ayu status                  # reports providers/credential presence, never secret values
ayu test                    # offline SQLite, memory, project/task and HTTPS-policy checks
ayu memory status
ayu project create my-app --root /path/to/existing/project
ayu tasks add my-app "Review open issues"
ayu project resume my-app
```

Configuration can be set in the process environment or a simple `.env` in the current working directory; process values override `.env`. The parser does not support shell expansion or multiline values. The tracked [`.env.example`](.env.example) contains placeholders only and is not loaded automatically. Copy values you need to a private `.env`, replace placeholders, and never commit it.

```dotenv
AYU_MODE=local
AYU_MEMORY_ENABLED=true
AYU_LOCAL_MODEL=/absolute/path/to/a-compatible-model.gguf
# Preferred backend: run a llama.cpp server bound to 127.0.0.1 first.
AYU_LOCAL_BACKEND=llama-server
AYU_LLAMA_SERVER_URL=http://127.0.0.1:9931
# Optional when the local server serves multiple model IDs.
# AYU_LOCAL_MODEL_ID=exact-id-from-v1-models
# Legacy backend only: AYU_LOCAL_BACKEND=llama-cli
# AYU_LLAMA_CLI=llama-cli
# AYU_LOCAL_MAX_INPUT_CHARS=65536
# AYU_DATABASE=~/.ayu-ai/ayu.sqlite3
# AYU_LOCAL_TIMEOUT=300
```

Set `AYU_MEMORY_ENABLED=false` to disable persistent-memory retrieval/writes in chat. Existing memories remain on disk, and conversation history is still persisted. Do not save credentials or other secrets as memories; saved user context may be sent to the explicitly selected inference provider.

## Local GGUF inference

The **recommended `llama-server` backend** uses llama.cpp's OpenAI-compatible chat-completions route, so message roles reach the runtime's model-specific chat template rather than being flattened by Ayu. It also asks that same server to count the templated input tokens and reads the effective context from `/props`; Ayu rejects a request whose input plus configured output budget exceeds that context. The backend requires a current llama.cpp server implementing `/health`, `/v1/models`, `/props`, `/v1/chat/completions/input_tokens`, and JSON-Schema-constrained chat responses.

Install/build llama.cpp separately, following the [official repository/build guide](https://github.com/ggml-org/llama.cpp) and [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). On Linux/macOS with CMake and a C++ compiler installed, a CPU build can be made with:

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$HOME/src/llama.cpp"
cmake -S "$HOME/src/llama.cpp" -B "$HOME/src/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/src/llama.cpp/build" --config Release --target llama-server -j 2
```

Model files are separate user-managed downloads; Ayu never downloads them. As a small reproducible example, the Qwen publisher lists the Qwen2.5-0.5B-Instruct Q4_K_M GGUF as Apache-2.0 and 491,400,032 bytes. Review the [model card/license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) before use. This immutable download URL and SHA-256 correspond to the publisher's listed file; download only if you explicitly choose this example:

```sh
mkdir -p "$HOME/models"
curl -L --fail --retry 3 \
  'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/6dd44a1fb35d11b5d1b28902876ce3cc9e882d0e/qwen2.5-0.5b-instruct-q4_k_m.gguf' \
  -o "$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
echo '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db  '"$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" | sha256sum --check
```

Choose a context size your RAM can handle; 4096 is a modest starting point, not a hardware guarantee. Start the server in a separate terminal and bind it only to loopback:

```sh
export AYU_LOCAL_MODEL="$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
"$HOME/src/llama.cpp/build/bin/llama-server" \
  -m "$AYU_LOCAL_MODEL" -c 4096 --host 127.0.0.1 --port 9931
```

In another terminal, configure Ayu and test it:

```sh
export AYU_MODE=local
export AYU_LOCAL_BACKEND=llama-server
export AYU_LOCAL_MODEL="$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
export AYU_LLAMA_SERVER_URL=http://127.0.0.1:9931
export AYU_MAX_OUTPUT_TOKENS=256
ayu models verify                 # file, server health, model ID/path, and context metadata only
ayu models verify --infer         # sends a short prompt and requires non-empty generated text
ayu chat --mode local --once "Explain what a context window is in one sentence."
```

`AYU_LOCAL_MODEL_ID` is optional for a single-model server, and required if multiple models are reported. Ayu accepts only a numeric loopback server URL (`127.0.0.0/8` or `::1`), disables proxy routing by using a direct socket, checks the GGUF path reported by `/props` against `AYU_LOCAL_MODEL`, and never contacts a remote provider from local mode. If the server reports that input plus output tokens exceed its actual context, shorten the prompt/history or restart it with a larger `-c`; Ayu does not silently truncate chat history. `AYU_LOCAL_MAX_INPUT_CHARS` (default 65,536; maximum 1,000,000) is an additional pre-network request-size guard, not a substitute for tokenizer-based context validation.

`ayu models verify` is only a preflight; it does **not** prove generation. `ayu models verify --infer` is the explicit inference request. The optional real-server integration test is skipped unless both `AYU_TEST_LLAMA_SERVER_URL` and `AYU_TEST_LLAMA_MODEL` point to a running loopback server and its model file. Fake-server tests verify only HTTP protocol/error behavior, not model loading or quality. The retained `llama-cli` backend continues to use a bounded prompt file and has process timeout/output limits, but it does not provide the server backend's chat-template token-count/context guarantee. Neither backend is an OS sandbox: the configured runtime/model parser runs with the current user's privileges. Runtime failure never triggers remote fallback.

PowerShell can set the same variables with `$env:AYU_MODE = "local"`, `$env:AYU_LOCAL_BACKEND = "llama-server"`, `$env:AYU_LOCAL_MODEL = "C:\models\model.gguf"`, and `$env:AYU_LLAMA_SERVER_URL = "http://127.0.0.1:9931"`. For a standalone `llama-cli` workflow, set `AYU_LOCAL_BACKEND=llama-cli` and `AYU_LLAMA_CLI` to the installed executable; the stronger context-aware chat check is specific to `llama-server`.

## Optional OpenAI, Anthropic, and NewsAPI

Choose **one** remote inference provider. The adapters use separate credential environment variables; keys are never shared between providers. Remote HTTP is rejected, error bodies are suppressed, and normal tests use mocks rather than paid/live calls.

OpenAI:

```sh
export AYU_MODE=remote
export AYU_PROVIDER=openai
export OPENAI_API_KEY="<your-key>"
export AYU_MODEL="<provider-supported-model-id>"
ayu config validate
ayu chat --mode remote --once "Hello"
```

Anthropic:

```sh
export AYU_MODE=remote
export AYU_PROVIDER=anthropic
export ANTHROPIC_API_KEY="<your-key>"
export ANTHROPIC_MODEL="<provider-supported-model-id>"
ayu config validate
ayu chat --mode remote --once "Hello"
```

For another OpenAI-compatible API, choose `AYU_PROVIDER=openai-compatible` (the default), set `AYU_API_KEY`, `AYU_BASE_URL=https://<provider-host>/v1`, and `AYU_MODEL`. Use HTTPS and the model/endpoint format documented by that provider.

NewsAPI is separate and does not provide LLM answers. Set `NEWSAPI_API_KEY` and `AYU_NEWSAPI_ENABLED=true`, then make each external request explicit:

```sh
export AYU_NEWSAPI_ENABLED=true
export NEWSAPI_API_KEY="<your-newsapi-key>"
ayu news search "climate policy" --from-date 2026-01-01 --sort-by relevancy --limit 5 --allow-network
```

The NewsAPI key is sent only to the fixed NewsAPI HTTPS endpoint in its API-key header. Article descriptions/URLs remain untrusted; verify them at the cited source. Respect the service's terms and rate limits. No live API call was made during development.

## Bounded read-only agent workflow

```sh
ayu tools list
ayu agents roles
ayu agents run "Find where the config timeout is validated" --workspace . --mode local --max-steps 3
ayu agents inspect RUN_ID
```

The workflow makes a bounded, sequential planner call, at most four allow-listed local tool calls, then analyst and reviewer calls. Current tools are literal text search and bounded UTF-8 source-file reading, confined to an explicitly supplied workspace; traversal, symlinks, hidden paths, secret/database extensions, unsupported file types, excess files/bytes, and invalid schemas are rejected. Search caps directories/files/bytes/matches and has a deadline; read caps each file at 512 KiB and each response at 150 lines. Registry time budgets reject late results but are not OS-level hard interrupts, so Ayu is not a filesystem sandbox. There is no shell, network, write, patch, arbitrary code/test execution, recursive loop, or tool-based privilege escalation. Run records/events are stored locally in the private SQLite database. The model reviewer is another model call, not independent proof.

Local mode requires a working local GGUF configuration. Remote mode sends the task and retrieved file excerpts to the selected remote provider, so it refuses to run unless you explicitly pass `--allow-remote-context`:

```sh
ayu agents run "Summarize the config timeout implementation" --workspace . --mode remote --allow-remote-context
```

Use that flag only when you consent to sending the selected excerpts. The read-only agent does not modify files or run tests; review its report yourself. Its multi-agent orchestration has been verified with deterministic provider mocks, not with a real model or a live remote provider.

## Bounded local coding workflow

`ayu code run` adds a narrowly scoped local coding loop and requires `AYU_LOCAL_BACKEND=llama-server` with JSON-Schema-constrained response support; `llama-cli` remains available for ordinary local chat but is not used for coding. It requires a clean Git repository root and creates a separate temporary Git worktree and task branch. The local model plans and reads at most the first 150 lines of each of up to three selected tracked text files. Each role uses strict JSON-Schema-constrained output; the workflow patches only inspected, existing implementation files, never test files, with up to five non-overlapping exact-context replacements and per-file atomic replacement plus best-effort rollback. Exact-context matching accepts LF or CRLF snippets against consistent LF/CRLF source files, preserves that file's newline convention, and rejects mixed or bare-CR source line endings. New-file creation is not part of this initial workflow. It runs one fixed Python test suite in a separately prepared, digest-pinned Docker image, permits at most one correction after a normal test failure, and asks a model reviewer to inspect the diff. It stores run status, hashes, the final diff, and verification evidence in the local audit database; test output is limited to hashed, bounded, best-effort-redacted excerpts (full output is not persisted). Nothing is committed or pushed; the returned worktree path is where the uncommitted result can be reviewed. A failure after run creation reports its `RUN_ID` so `ayu code inspect RUN_ID` can recover the audit events and worktree path. Interrupted work remains unverified and is not automatically resumed.

```sh
# Start llama-server as described above, then set the same local configuration:
export AYU_MODE=local
export AYU_LOCAL_BACKEND=llama-server
export AYU_LOCAL_MODEL="$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
export AYU_LLAMA_SERVER_URL=http://127.0.0.1:9931
export AYU_MAX_OUTPUT_TOKENS=512

# Configure AYU_SANDBOX_IMAGE to a locally available immutable image digest first.
ayu sandbox status
# Choose a fixed Python suite. Ayu refuses to proceed unless the local Docker daemon and image pass preflight.
ayu code run "Fix the failing calculator addition test" --workspace /path/to/clean/repo \
  --test-suite pytest --test-path tests
ayu code inspect RUN_ID
ayu sandbox cleanup RUN_ID  # explicit recovery: remove only containers recorded for this Ayu run
```

The operator must prepare the image separately, install the selected test tools into it, inspect its provenance, and set `AYU_SANDBOX_IMAGE=repository@sha256:<64-lowercase-hex-digest>`. Ayu does not build or pull images, install dependencies, or use unpinned tags during a coding run. Optional `AYU_DOCKER_BINARY` and `AYU_DOCKER_HOST` may select a Docker CLI/Unix socket; remote TCP endpoints and user Docker contexts are rejected, and configured POSIX socket paths must exist as sockets. A socket can still be a user-managed proxy to another daemon, so its locality and trustworthiness remain the operator's responsibility. `ayu sandbox status` verifies daemon/image preflight without running project code. A container name is persisted before launch; `ayu sandbox cleanup RUN_ID` is an explicit recovery operation that force-removes only Ayu-generated containers listed in that run's audit. This is not automatic resume and cannot recover a host/daemon failure without a working Docker service. Without Docker, an image digest, or a ready local daemon, the coding CLI fails closed before provider/model initialization; it never falls back to host test execution.

Supported suites are `pytest`, `unittest`, and `compileall`; model-generated shell commands are never executed. Test paths must be workspace-relative; safe directories such as `tests/` and allow-listed text-source files are accepted, while hidden, denied, and symlink paths are rejected. Each container requests a 120-second default timeout (maximum 600), 512 MiB memory, one CPU, 64 PIDs, 64 MiB `/tmp`, bounded output, no network, a non-root UID, dropped Linux capabilities, `no-new-privileges`, a read-only root filesystem, and a read-only disposable workspace snapshot. The snapshot excludes `.git` metadata, selected hidden configuration, known secret/credential names and extensions, and common credential directories; symlink targets are refused, hard-linked files and nested mount points are omitted, and streaming snapshot preparation is limited to 20,000 entries, 10,000 files, 256 MiB total, 32 MiB per file, and a cooperative 60-second deadline between filesystem operations. A blocked kernel/filesystem call can exceed that deadline. Image-defined environment variables are cleared unless Ayu sets a sanitized value. These are requested layered controls, not a proof against every secret filename, malicious image, container/kernel escape, or daemon compromise. Docker and the host kernel remain trusted; rootful Docker daemon compromise can affect the host. Container cleanup is requested and verified when possible on timeout/cancellation; interrupted/ambiguous runs are not automatically resumed. Ayu refuses dirty source worktrees, configured Git filters/hooks, path traversal, symlink edits, stale/ambiguous contexts, and unsupported file types. Coding inference is local-only in this initial implementation; no coding context is sent to a remote provider.

The earlier real-model smoke run in this repository predates the Docker sandbox integration and executed a test on the host; it is not evidence that the new sandbox backend works. The sandbox policy unit tests use mocks and source-level command inspection only. In this development environment Docker/Podman/bubblewrap are unavailable, the Docker socket is absent, and creating a network namespace is denied by the environment; therefore actual filesystem/network/resource isolation is **BLOCKED / NOT VERIFIED** here, and the integrated `ayu code run` path cannot currently complete in this environment. The previous real-model patch success and later exact-context safe refusal are evidence only for the pre-sandbox coding loop, not the current isolated execution boundary. Windows/macOS runtime isolation, cancellation/orphan cleanup under process crash, and general coding-model correctness remain unverified. Model review is not independent proof; inspect the diff and test evidence yourself before accepting it. See [the coding-agent threat model](docs/SECURITY_THREAT_MODEL.md) for assets, assumptions, residual risks, and verification labels.

## Memory, projects, and recovery

```sh
ayu memory add --category preference "Prefer concise answers"
ayu memory search concise --category preference
ayu memory show 1
ayu memory update 1 --text "Prefer concise technical answers"
ayu memory delete 1
ayu project create notes --root ./notes
ayu tasks add notes "Check backup"
ayu tasks add notes "Write backup report" --depends-on 1
ayu tasks start 1
ayu tasks complete 1
ayu tasks inspect 2
ayu tasks set-dependencies 2 --depends-on 1
ayu project inspect notes
ayu project resume notes
ayu project update notes --field goals --value '["Ship a tested CLI"]'
ayu memory export ./memories.json
ayu memory import ./memories.json
ayu backup ./ayu-backup.sqlite3
ayu restore ./ayu-backup.sqlite3 --confirm
```

Memory categories are `working`, `episodic`, `semantic`, `project`, and `preference`. SQLite records have timestamps/provenance; retrieval is lexical, not vector search. Task prerequisites are limited to existing tasks in the same project; updates reject cycles, and `in_progress`/`completed` transitions are blocked until every prerequisite is complete. Completed tasks cannot be reopened; create a new task to preserve history. Project resume selects a ready task and reports blocked prerequisites; it does not autonomously claim completion. Restore validates a staged database, creates an automatic pre-restore recovery copy, and requires explicit `--confirm`. Close other Ayu processes before restore. Local databases and exports use owner-only POSIX permissions where supported; Windows ACL behavior follows the host directory policy.

## Local document search

```sh
ayu docs search ./my-documents "search phrase" --limit 20
```

This is a bounded lexical scan of selected text formats (TXT, Markdown, source, JSON, CSV, TOML, YAML, HTML, CSS, shell and PowerShell). It does not upload files or provide AI summaries/citations. It is separate from the model-assisted agent tools; both require explicit local paths/workspaces.

## Native AyuLM: train, evaluate, generate, benchmark

AyuLM uses a reversible UTF-8 byte tokenizer, token embeddings, rotary positional encoding, causal multi-head attention, feed-forward blocks, normalization, tied output weights, autoregressive sampling, and next-byte cross-entropy. Train only with text you are entitled to use. The implementation does not validate dataset licenses.

```sh
python -m pip install -e '.[training]'
ayu test --ayulm-smoke
# Replace the corpus path with text you own or are licensed to use.
ayu train --data ./licensed-or-owned-corpus.txt --output ./ayu-small.pt \
  --steps 200 --seq-len 64 --d-model 64 --heads 4 --layers 2 --seed 42
ayu checkpoint inspect ./ayu-small.pt
ayu evaluate ./ayu-small.pt --data ./heldout-text.txt
ayu generate ./ayu-small.pt "Once upon a time" --tokens 80
ayu benchmark ./ayu-small.pt --tokens 32 --repeats 3
ayu train-resume ./ayu-small.pt --data ./licensed-or-owned-corpus.txt --steps 100
```

The native pipeline uses a deterministic, disjoint internal train/validation byte split, saves optimizer/RNG state for resumption, checks corpus identity/configuration when resuming, and writes tokenizer metadata and a local model registry entry under `~/.ayu-ai/` by default. External `evaluate` data should be held out by the user. Benchmark reports local autoregressive throughput, not quality or a cross-machine score. Generated samples can be incoherent or contain replacement characters until a sufficiently large, appropriate corpus and training budget are used. Model files may be large and checkpoints record source paths; inspect metadata before sharing.

## Tests and platform support

```sh
python -m pip install -e '.[dev]'
python -m compileall -q ayu tests
python -m pytest -q
ayu test
ayu test --ayulm-smoke  # optional PyTorch required
```

Ordinary tests are offline and require no real credentials. Provider/API behavior is mocked. The optional AyuLM integration test uses a tiny synthetic corpus. The previously published feature commit `363b3b9` passed [Actions run 40341486036](https://github.com/teamhecked-Ayu/Ayu-AI/actions/runs/40341486036) on 2026-10-26: nine core jobs covered Ubuntu, Windows, and macOS with Python 3.10, 3.11, and 3.12, and the CPU AyuLM train/checkpoint/generation job passed. The separate [Gitleaks run 40341486052](https://github.com/teamhecked-Ayu/Ayu-AI/actions/runs/40341486052) also passed. These checks predate the Docker sandbox changes in the current worktree and must not be treated as their verification; CI uses deterministic/fake providers and does not verify a live local model on each OS. The development/verification environment used for direct runtime checks was Linux (Ubuntu sandbox), Python 3.12.3, CPU, and PyTorch 2.14.1+cpu. Windows and macOS were CI-tested on the previous published commit, not manually tested. ARM, Android/Termux, CUDA, and live provider APIs were not tested.

## Implemented scope and limitations

Implemented (application policy; unit-tested with mocks): no-key basic CLI; persistent private local SQLite memory/project/task/chat state, same-project task dependencies, and backup/restore; local GGUF preflight/process adapter; optional OpenAI, Anthropic, and separate NewsAPI clients; bounded search/read tool registry; sequential planner/researcher/analyst/reviewer orchestration with schema/path checks and local audit events; local-only coding workflow with clean-repository checks, temporary Git worktrees, exact-context atomic patches, fixed allow-listed Python verification, one correction, persisted review/diff evidence, and a fail-closed Docker runner configured for a digest-pinned image, no network, read-only snapshot/root, non-root UID, resource bounds, and auditable container cleanup; AyuLM native training, resumption, evaluation, generation and benchmark; offline tests and CI definitions.

Not verified/blocked: real Docker filesystem/network/resource isolation, container escape resistance, image provenance, cancellation and cleanup under process/host crash, rootless Docker/Desktop semantics, and coding runtime on any platform; the current sandbox has no Docker CLI/daemon or image and cannot create a network namespace. The implementation therefore refuses to run coding tests here. Orphan cleanup is available only for container names persisted by Ayu; a crash before the startup audit event or host/daemon failure may still need operator investigation. Automatic crash recovery/restart of a coding run is not implemented. Also not implemented: remote coding mode; arbitrary shell or user-defined test commands; non-Python coding/test runners; crash-safe multi-file transaction across power loss; cross-project/advanced workflow orchestration; browser/MCP servers; vector embeddings; automatic model download/catalogue; full article retrieval; cancellable remote requests; encrypted database storage; multi-user access control; or production-service guarantees. Termux remains documented, not verified on an Android device.

## Upgrade and uninstall

From a source checkout, update with `git pull` and `python -m pip install --upgrade -e .` in the active environment; review changes and back up the database first. `python -m pip uninstall ayu-ai` removes the package, not local user data. Remove `~/.ayu-ai/` only when you intentionally want to delete the database, registry, and any checkpoints stored there.
