# Android / Termux device verification

This procedure validates the **basic Python + SQLite CLI profile** on a real
Android device running Termux. It does not certify local LLM inference, GPU
acceleration, Docker sandboxing, background execution, or all Android versions.

## Install the baseline

Use the official Termux distribution. Install prerequisites with Termux's
package manager, then clone the repository:

```sh
pkg update
pkg install python git
git clone https://github.com/teamhecked-Ayu/Ayu-AI.git
cd Ayu-AI
bash scripts/termux-install.sh
```

The installer checks Python 3.10+, creates or reuses a project-local `.venv`,
installs Ayu inside that environment, and runs the offline smoke test. It does
not install packages into Termux's managed Python, upgrade pip, download model
weights, or configure an inference server. If an existing `.venv` is malformed,
it stops rather than deleting or replacing it.

If pip reports a build/backend incompatibility, record the exact Python, pip,
setuptools and wheel versions. Do not disable package isolation or bypass
Termux package safeguards as a blind workaround.

## Run the on-device smoke test

The installer already runs the smoke test. To repeat it later:

```sh
. .venv/bin/activate
bash scripts/termux-smoke.sh
ayu doctor --json
```

The smoke script checks the installed CLI entry point, Ayu's offline self-test,
diagnostics JSON/privacy invariants, and a temporary SQLite database lifecycle.
It writes only to a temporary directory and removes that directory on exit.
Do not run it while pointing `AYU_DATABASE` at an existing personal database.

## Record actual device evidence

For a useful compatibility report, record:

- Android version and device architecture (`uname -m`)
- Termux source/distribution and `termux-info` output (review before sharing)
- `python --version`, `python -m pip --version`, and `sqlite3.sqlite_version`
- Whether installation completed without modifying Termux's managed Python
- The complete smoke-test result and `ayu doctor --json` output after checking it for private paths
- Available storage and whether the process survives screen-off / Android background limits

Do not publish usernames, home-directory paths, device identifiers, tokens, or
other private values in issue reports.

## Verification boundary

The CI job named **Termux compatibility contract (Linux simulation)** supplies
Termux-like environment markers on Ubuntu. It validates the script and
platform-detection contract, but it is not an Android emulator and does not
prove Termux compatibility. Only a successful run on a real device counts as
manual Termux evidence.

Local model servers and model weights are a separate milestone. A passing
baseline smoke test means only that the basic CLI and SQLite path worked.
