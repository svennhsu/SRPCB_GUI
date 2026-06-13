#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    source ".venv/bin/activate"
elif [ -d ".venv-gpu" ]; then
    source ".venv-gpu/bin/activate"
fi

export PYTHONUNBUFFERED=1

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "Error: Python not found. Install python3." >&2
    exit 1
fi
exec "$PYTHON" app/main.py "$@"
