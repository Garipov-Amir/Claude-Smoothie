#!/usr/bin/env bash
# Run a Blender generation script headless.
# Usage: run_blender.sh <script.py> [arg1 arg2 ...]   (args are forwarded to the script's sys.argv, after a single "--")
set -euo pipefail

BLENDER_BIN="${BLENDER_BIN:-}"
if [ -z "$BLENDER_BIN" ]; then
  if command -v blender >/dev/null 2>&1; then
    BLENDER_BIN="$(command -v blender)"
  elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
  fi
fi

if [ -z "$BLENDER_BIN" ]; then
  # no Blender binary: fall back to a python that has the bpy wheel
  # (pip install bpy) — same API, used on CI / cloud containers
  PY="${BPY_PYTHON:-python3}"
  if "$PY" -c "import bpy" >/dev/null 2>&1; then
    [ "$#" -lt 1 ] && { echo "usage: run_blender.sh <script.py> [args...]" >&2; exit 1; }
    SCRIPT="$1"; shift
    exec "$PY" "$SCRIPT" -- "$@"
  fi
  echo "error: blender not found. Install with: brew install --cask blender (or pip install bpy)" >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "usage: run_blender.sh <script.py> [args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift

exec "$BLENDER_BIN" --background --factory-startup --python "$SCRIPT" -- "$@"
