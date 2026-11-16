#!/usr/bin/env bash
# Offline baseline smoke test for an installed Ayu CLI on Termux or Linux CI.
set -euo pipefail

if ! command -v ayu >/dev/null 2>&1; then
  echo "ERROR: ayu command not found. Install the project in the active Python environment first." >&2
  exit 2
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/ayu-termux-smoke.XXXXXX")"
cleanup() { rm -rf "$tmp_root"; }
trap cleanup EXIT

export AYU_MODE=basic
export AYU_MEMORY_ENABLED=false
export AYU_DATABASE="$tmp_root/ayu.sqlite3"

echo "[1/4] CLI entry point"
ayu --version

echo "[2/4] SQLite and offline self-test"
ayu test

echo "[3/4] Privacy-safe platform diagnostics"
doctor_json="$(ayu doctor --json)"
AYU_DOCTOR_JSON="$doctor_json" python - <<'PY'
import json
import os

report = json.loads(os.environ["AYU_DOCTOR_JSON"])
assert report.get("schema_version") == "ayu-platform-diagnostics-v1"
assert report.get("platform", {}).get("profile") == "android-termux", report.get("platform", {}).get("profile")
assert report.get("privacy", {}).get("network_requests") == 0
assert report.get("runtime", {}).get("server_contacted") is False
assert report.get("status") in {"ready", "warning"}, report.get("status")
print("diagnostics schema and offline/privacy assertions passed")
PY

echo "[4/4] Temporary database lifecycle"
test -f "$AYU_DATABASE"
echo "Ayu baseline smoke test passed. Temporary data will now be removed."
