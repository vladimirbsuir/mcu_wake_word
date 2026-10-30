#!/usr/bin/env bash
# Install Node and Python dependencies for WakeForge Studio.
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/setup.sh --recreate
#   PYTHON=/opt/python3.12/bin/python3 bash scripts/setup.sh
#
# Options:
#   --recreate    Delete and rebuild .venv.
#   --skip-node   Do not run npm install.
#   -h, --help    Show this help.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
VENV_PYTHON="$VENV/bin/python"
REQUIREMENTS="$ROOT/python/requirements.txt"
CHECK_ENV="$ROOT/python/check_env.py"
RECREATE=0
SKIP_NODE=0
PYTHON_OVERRIDE="${PYTHON:-}"

for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    --skip-node) SKIP_NODE=1 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

step() { printf '\033[36m==> %s\033[0m\n' "$1"; }
ok() { printf '\033[32mOK  %s\033[0m\n' "$1"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$1"; }

step "WakeForge Studio setup"
echo "    Project: $ROOT"

if [[ "$SKIP_NODE" -eq 0 ]]; then
  if ! command -v node >/dev/null 2>&1; then
    echo "Node.js 20+ was not found. Install it from https://nodejs.org and re-run." >&2
    exit 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm was not found. Reinstall Node.js with npm included." >&2
    exit 1
  fi
  step "Installing frontend dependencies (Node $(node --version))"
  npm install
  ok "Node dependencies installed."
fi

find_python() {
  if [[ -n "$PYTHON_OVERRIDE" ]]; then
    if ! command -v "$PYTHON_OVERRIDE" >/dev/null 2>&1; then
      echo "PYTHON=$PYTHON_OVERRIDE was not found on PATH." >&2
      return 1
    fi
    echo "$PYTHON_OVERRIDE"
    return 0
  fi

  local candidate version
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
      case "$version" in
        3.10|3.11|3.12)
          echo "$candidate"
          return 0
          ;;
      esac
    fi
  done
  return 1
}

if [[ "$RECREATE" -eq 1 && -d "$VENV" ]]; then
  warn "Removing the existing .venv"
  rm -rf "$VENV"
fi

if [[ -x "$VENV_PYTHON" ]]; then
  step "Reusing .venv"
else
  PYTHON_BIN="$(find_python)" || {
    echo "Python 3.10-3.12 was not found. Install it and re-run, or set PYTHON=<path>." >&2
    exit 1
  }
  step "Creating .venv with $PYTHON_BIN (TensorFlow requires Python 3.10-3.12)"
  "$PYTHON_BIN" -m venv "$VENV"
  ok "Virtual environment created."
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo ".venv is incomplete. Re-run with --recreate." >&2
  exit 1
fi

step "Installing Python ML dependencies ($("$VENV_PYTHON" -c 'import platform; print(platform.python_version())'))"
"$VENV_PYTHON" -m pip install --upgrade pip wheel setuptools
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"

PROBE="$("$VENV_PYTHON" "$CHECK_ENV" | tail -n 1)"
if ! PROBE="$PROBE" "$VENV_PYTHON" - <<'PY'
import json
import os
import sys

info = json.loads(os.environ["PROBE"])
if not info.get("supported_python"):
    sys.exit(f"Python {info.get('version')} is not supported; use 3.10-3.12.")
missing = [name for name, present in info.get("packages", {}).items() if not present]
if missing:
    sys.exit("Missing packages: " + ", ".join(sorted(missing)))
PY
then
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  warn "Rust/cargo was not found. Install Rust from https://rustup.rs to run the desktop app."
fi
if ! command -v npm >/dev/null 2>&1; then
  warn "npm was not found. Install Node.js before running the app."
fi

echo
ok "Setup complete."
echo "    Python : $VENV_PYTHON"
echo "    Start  : npm run tauri dev"
echo
echo "If the app uses a different interpreter, set WAKE_WORD_PYTHON to its path."
