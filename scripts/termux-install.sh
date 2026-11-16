#!/usr/bin/env bash
# Create an isolated project venv and install Ayu without changing Termux's managed Python.
set -euo pipefail

if [[ -z "${TERMUX_VERSION:-}" && "${PREFIX:-}" != *"/com.termux/"* ]]; then
  echo "ERROR: this installer is intended for Termux. Run it inside the Termux app." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f pyproject.toml || ! -d ayu ]]; then
  echo "ERROR: repository root is incomplete; pyproject.toml and ayu/ are required." >&2
  exit 2
fi

python_bin="${PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "ERROR: Python was not found. Install it with Termux's package manager, then retry." >&2
  exit 2
fi

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Ayu requires Python 3.10 or newer.")
print(f"Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY

venv_dir="$repo_root/.venv"
if [[ -e "$venv_dir" && ! -x "$venv_dir/bin/python" ]]; then
  echo "ERROR: .venv exists but is not a valid Unix Python virtual environment; inspect it manually." >&2
  exit 2
fi
if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi

"$venv_dir/bin/python" -m pip install .
export PATH="$venv_dir/bin:$PATH"

echo "Running Ayu's offline Termux baseline checks..."
bash scripts/termux-smoke.sh

echo "Installation and baseline checks completed."
echo "Activate later with: source .venv/bin/activate"
