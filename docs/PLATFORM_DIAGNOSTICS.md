# P5: Platform diagnostics

P5 adds `ayu doctor --json`, a machine-readable, privacy-safe platform report intended to make desktop and Android/Termux support measurable rather than assumed.

## Run

```sh
ayu doctor
ayu doctor --json
```

The JSON report uses schema `ayu-platform-diagnostics-v1`. It records the operating-system profile, CPU architecture, Python implementation/version, SQLite version, CPU count, selected local backend, basic executable/dependency detection, free disk space, and bounded local readiness checks.

On Android, it attempts to distinguish a general Android Python environment from Termux using platform/runtime markers. It does not print the environment values used for detection.

## Privacy and side effects

- Makes zero network requests and never contacts a local or remote model server.
- Does not include credential values, configured model names, database paths, or absolute filesystem paths.
- Uses a temporary file to probe database-directory write access and removes it immediately.
- Does not create model files, download packages, start Docker, or change configuration.
- Reports only whether optional PyTorch, NumPy, and Docker executables are detected.

## Interpreting results

- `ready`: no blocking check failed and no warning was recorded.
- `warning`: a non-blocking capability is missing, such as an unconfigured local model or unavailable `llama-cli`.
- `fail`: Python is below the supported minimum or the database directory is unavailable/not writable.

For server-backed llama.cpp, Ollama, and vLLM, the report explicitly says the endpoint is configured but not contacted. Docker executable detection is not a Docker daemon health check.

## Verification boundary

The report is a diagnostic snapshot, not a platform certification. It cannot prove Android/Termux package compatibility, native inference availability, GPU acceleration, available runtime memory, sandbox isolation, model provenance, or answer quality. A real Android/Termux device test is still required before claiming that platform is supported.
