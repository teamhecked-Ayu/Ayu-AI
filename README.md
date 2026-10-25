# Ayu AI / AyuLM

Ayu AI is a terminal-first, local-first **foundation**. It is intentionally useful without a paid API key and does not silently fall back to a remote model. This repository is not a pretrained conversational model, a production autonomous operating system, or a competitor to leading AI labs. Its native AyuLM model is a small, educational byte-level decoder Transformer that can be trained from scratch with user-supplied text and optional PyTorch.

## Operating modes

- **Basic:** CLI, local SQLite memory, project/task bookkeeping, configuration checks, and lexical text-document search. No language-model reasoning is simulated.
- **Local:** explicit local inference through an installed `llama.cpp` `llama-cli` executable and a user-provided compatible GGUF model. The application does not download models or contact remote providers in this mode.
- **Remote (optional):** OpenAI-compatible API adapter, used only when explicitly requested with `ayu chat --mode remote` or `AYU_MODE=remote`. Chat content is sent to the configured provider.
- **AyuLM research:** native model training, checkpointing, evaluation, and generation; optional PyTorch dependency. A newly trained tiny model is not a pretrained assistant.

## Install

Requires Python 3.10+. From a checkout, create an isolated environment:

### Linux / macOS

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ayu --help
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
ayu --help
```

### Android Termux

Install Python using Termux's package manager (`pkg install python`), then use the Linux steps. Python CPU-only basic commands are the intended baseline. PyTorch wheels and llama.cpp builds may not be available or practical on every Android/ARM device; do not assume GPU acceleration, adequate RAM, or that a model will fit. No model weights are bundled.

For native AyuLM training only, install the optional extra where a compatible PyTorch wheel exists:

```sh
python -m pip install -e '.[training]'
```

## First-run basics

```sh
ayu                         # help; no API key required
ayu init                    # create a private local .env template; never overwrites by default
ayu doctor                  # local checks, no network request
ayu status                  # redacts the API key value
ayu memory status
ayu project create my-app --root /path/to/existing/project
ayu tasks add my-app "Review open issues"
ayu project resume my-app
```

Configuration variables can be set in the environment or a simple `.env` in the current working directory. Process environment values override `.env`. The `.env` parser does not support shell expansion or multiline values.

```dotenv
AYU_MODE=local
AYU_MEMORY_ENABLED=true
AYU_LOCAL_MODEL=/absolute/path/to/a-compatible-model.gguf
AYU_LLAMA_CLI=llama-cli
# Optional remote adapter only:
# AYU_MODE=remote
# AYU_API_KEY=your-provider-key
# AYU_BASE_URL=https://api.openai.com/v1
# AYU_MODEL=provider-supported-model-id
# AYU_DATABASE=~/.ayu-ai/ayu.sqlite3
# AYU_LOCAL_TIMEOUT=300
```

Set `AYU_MEMORY_ENABLED=false` to disable saved-memory retrieval in chat and reject memory writes. Existing records remain on disk until explicitly deleted; this does not disable conversation-history persistence. Configuration examples are in [`examples/ayu.env.example`](examples/ayu.env.example).

Keep `.env` private and never commit credentials. `ayu init` sets `.env` permissions to owner-only where the platform supports it. Remote calls require HTTPS; plaintext HTTP and URL-embedded credentials are rejected. Do not save secrets in memory: stored memory may be included in model context. SQLite database files and exports are created with owner-only POSIX permissions where supported; Windows ACL behavior depends on the user's directory policy.

## Local model setup

Install `llama.cpp` yourself using its official platform instructions, obtain a GGUF model from a source you trust, and review the model's license and hardware requirements. Ayu will not choose or download a model for you. Set `AYU_LOCAL_MODEL` and, if needed, `AYU_LLAMA_CLI` to the executable path.

```sh
ayu models info
ayu models verify
ayu chat --mode local
ayu chat --mode local --role coder --once "Explain this error: ..."
```

`models verify` checks that the configured file exists, has a `.gguf` suffix, and a llama.cpp executable can be found. This is a preflight check; it does not prove that every GGUF architecture/quantization is supported or that the model fits available RAM. Inference attempts report runtime errors instead of falling back remotely.

## Memory and project continuity

```sh
ayu memory add --category preference "Prefer concise answers"
ayu memory search concise --category preference
ayu memory show 1
ayu memory update 1 --text "Prefer concise technical answers"
ayu memory delete 1
ayu project create notes --root ./notes
ayu tasks add notes "Check backup"
ayu project inspect notes
ayu project resume notes
ayu project update notes --field goals --value '["Ship a tested CLI"]'
```

Memory categories are `working`, `episodic`, `semantic`, `project`, and `preference`. Records are local SQLite rows, tagged with timestamps and provenance. Duplicate content in the same category/project is updated rather than duplicated. Retrieval is simple lexical overlap; no hosted embeddings are used. Export/import uses validated JSON; `AYU_MEMORY_ENABLED=false` disables retrieval and new writes but does not delete old records or stop chat-history persistence.

```sh
ayu memory export ./memories.json
ayu memory import ./memories.json
ayu backup ./ayu-backup.sqlite3
ayu restore ./ayu-backup.sqlite3 --confirm
```

Restore is destructive and requires `--confirm`; it validates a staged copy, makes a private automatic pre-restore recovery copy, then replaces the active database. Close other Ayu processes before restoring. Project resume checks saved references and lists the next pending task; it does not autonomously continue work or claim an unverified task succeeded.

## Local document search

```sh
ayu docs search ./my-documents "search phrase" --limit 20
```

Search is a bounded lexical scan of selected text formats (TXT, Markdown, source code, JSON, CSV, TOML, YAML, HTML, CSS, shell and PowerShell). It does not upload files, parse every possible document format, or produce AI summaries/citations. The root must be supplied explicitly.

## Native AyuLM research (optional PyTorch)

AyuLM-native uses a reversible UTF-8 byte tokenizer (260 token IDs), token embeddings, rotary positional encoding, causal multi-head self-attention, feed-forward blocks, normalization, tied output projection, autoregressive sampling, and next-byte cross-entropy training. It is an actual small-model research implementation, not trained weights. Train only on corpora you are entitled to use; the software does not check dataset license or provenance.

```sh
python -m pip install -e '.[training]'
# Supply a local text corpus large enough for the chosen context length.
ayu train --data ./licensed-or-owned-corpus.txt --output ./ayu-small.pt \
  --steps 200 --seq-len 64 --d-model 64 --heads 4 --layers 2 --seed 42
ayu checkpoint inspect ./ayu-small.pt
ayu evaluate ./ayu-small.pt --data ./heldout-text.txt
ayu generate ./ayu-small.pt "Once upon a time" --tokens 80
ayu train-resume ./ayu-small.pt --data ./licensed-or-owned-corpus.txt --steps 100
```

The training pipeline uses a deterministic, disjoint byte-level train/validation split (at least `2 * (seq_len + 2)` bytes), saves model/optimizer/RNG state for resumption, validates the training corpus hash and model configuration when resuming, reports loss/perplexity, writes tokenizer metadata, and records a local model registry entry under `~/.ayu-ai/model-registry.json` by default. Use the same corpus when resuming. These are next-token metrics, not conversational benchmarks. Tiny corpora and tiny models typically produce poor, nonsensical generations. Training and checkpoint loading require the optional PyTorch package and adequate memory; GPU use is optional and device-specific. Checkpoints contain user-data provenance paths; do not publish them without reviewing metadata.

## Commands and tests

```sh
python -m pip install -e '.[dev]'
ayu --help
ayu chat --help
ayu models --help
ayu memory --help
ayu project --help
ayu tasks --help
python -m pytest -q
```

The default install includes NumPy for the retained compatibility prototype. Optional PyTorch is needed only for native AyuLM training. Basic commands do not need an API key, external database, GPU, Docker, or network access. Remote inference is opt-in; no hidden telemetry is implemented.

Upgrade a source checkout with `git pull` followed by `python -m pip install --upgrade -e .` inside the active environment; review release notes and back up the local SQLite database first.

## Implemented versus not implemented

Implemented and tested without network: command parsing, SQLite chat/memory/project/task storage, safe credential redaction in status/config output, local document lexical search, remote adapter request handling under mocked tests, and local llama.cpp preflight/process adapter. Optional native AyuLM architecture/training code is present; this environment has no PyTorch, so actual optimization, checkpoint round trips, and generated text have **not** been runtime-verified here.

Not implemented: agent orchestration, autonomous task execution, shell/code execution, patch application, browser or external research, vector embeddings, project file indexing, online model catalogue/downloads, migration tooling beyond additive SQLite schema upgrades, cancellation during local inference, encrypted storage, multi-user access control, production service guarantees, and comprehensive Windows/Termux integration testing. Contributions should add these only with explicit safety policies and tests.

## Uninstall

Uninstall the package from the active environment with `python -m pip uninstall ayu-ai`. This does not delete user data. Remove `~/.ayu-ai/` only if you intentionally want to erase the local database, registry, and checkpoints stored there.
