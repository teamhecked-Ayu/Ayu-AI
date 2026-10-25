# Ayu AI / AyuLM

Ayu AI is a terminal-first, local-first software foundation. It is useful without API credentials and never silently falls back to a remote model. AyuLM is a small byte-level decoder Transformer for educational/research training—not a pretrained assistant or a claim of frontier-level capability. The software does not claim autonomous production-OS guarantees.

## Operating modes and optional services

- **Basic:** local CLI, SQLite conversation/memory/project/task records, configuration checks, and bounded lexical text search. It does not pretend to provide LLM reasoning.
- **Local inference:** an explicitly selected, already-installed `llama.cpp` `llama-cli` and a user-supplied compatible GGUF. Ayu does not download weights or contact an AI provider in local mode.
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
AYU_LLAMA_CLI=llama-cli
# AYU_DATABASE=~/.ayu-ai/ayu.sqlite3
# AYU_LOCAL_TIMEOUT=300
```

Set `AYU_MEMORY_ENABLED=false` to disable persistent-memory retrieval/writes in chat. Existing memories remain on disk, and conversation history is still persisted. Do not save credentials or other secrets as memories; saved user context may be sent to the explicitly selected inference provider.

## Local GGUF inference

Install/build `llama-cli` from the upstream llama.cpp project using instructions suitable for your operating system, obtain a compatible GGUF from a source you trust, and review its license and hardware requirements. Ayu does not choose/download weights. Configure paths explicitly:

```sh
export AYU_MODE=local
export AYU_LOCAL_MODEL="$HOME/models/model.gguf"
export AYU_LLAMA_CLI="$(command -v llama-cli)"
ayu models verify
ayu chat --mode local
ayu chat --mode local --role coder --once "Explain this error: ..."
```

PowerShell equivalent:

```powershell
$env:AYU_MODE = "local"
$env:AYU_LOCAL_MODEL = "$HOME\models\model.gguf"
$env:AYU_LLAMA_CLI = "llama-cli"
ayu models verify
ayu chat --mode local
```

`models verify` checks a nonempty/readable GGUF path and runtime executable; it cannot guarantee that a particular architecture/quantization is supported or that the model fits RAM. Local runs have a configured wall-time limit and cap captured stdout/stderr at 4 MiB per stream; runtime diagnostics are suppressed because they may echo prompt content. The configured binary still runs with your OS user's privileges—it is not sandboxed. No real GGUF/llama.cpp inference was available for verification in the development sandbox. Runtime failure never triggers remote fallback.

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

Use that flag only when you consent to sending the selected excerpts. The agent does not modify files or run tests; review its report yourself. End-to-end agent behavior has been verified with deterministic provider mocks, not a real remote model or an available GGUF model.

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

Ordinary tests are offline and require no real credentials. Provider/API behavior is mocked. The optional AyuLM integration test uses a tiny synthetic corpus. The latest observed GitHub Actions core matrix completed successfully on Linux, Windows, and macOS with Python 3.10, 3.11, and 3.12 (nine jobs); the separate secret scan also passed. The hosted CPU AyuLM job in that same latest run was still in progress when this note was written (it passed in an earlier run); check the linked commit's Actions page before relying on a later result. The development/verification environment used for this release was Linux (Ubuntu sandbox), Python 3.12.3, CPU, and PyTorch 2.14.1+cpu. Windows and macOS were CI-tested, not manually tested. ARM, Android/Termux, CUDA, and live provider APIs were not tested.

## Implemented scope and limitations

Implemented: no-key basic CLI; persistent private local SQLite memory/project/task/chat state, same-project task dependencies, and backup/restore; local GGUF preflight/process adapter; optional OpenAI, Anthropic, and separate NewsAPI clients; bounded search/read tool registry; sequential planner/researcher/analyst/reviewer orchestration with schema/path checks and local audit events; AyuLM native training, resumption, evaluation, generation and benchmark; offline tests and CI definitions.

Not implemented: write/patch application, arbitrary shell or code execution, test-running agents, cross-project/advanced workflow orchestration, browser/MCP servers, vector embeddings, automatic model download/catalogue, full article retrieval, cancellable remote requests, encrypted database storage, multi-user access control, or production-service guarantees. These are deliberately not represented as working features. Termux remains documented, not verified on an Android device.

## Upgrade and uninstall

From a source checkout, update with `git pull` and `python -m pip install --upgrade -e .` in the active environment; review changes and back up the database first. `python -m pip uninstall ayu-ai` removes the package, not local user data. Remove `~/.ayu-ai/` only when you intentionally want to delete the database, registry, and any checkpoints stored there.
