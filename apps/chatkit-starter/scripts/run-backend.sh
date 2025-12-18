#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_NAME="${1:-backend}"
BACKEND_DIR="${APP_ROOT}/${BACKEND_NAME}"
PORT="${PORT:-8000}"

cd "$BACKEND_DIR"

# Detect backend type and run accordingly
if [[ -f "deno.json" ]]; then
    # TypeScript/Deno backend
    export PATH="$HOME/.deno/bin:$PATH"
    exec deno run --allow-net --allow-env --allow-read --unstable-sloppy-imports --watch main.ts
elif [[ -f "pyproject.toml" ]]; then
    # Python backend
    VENV_DIR="${BACKEND_DIR}/.venv"
    UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --directory "${BACKEND_DIR}"
    exec "${VENV_DIR}/bin/python" -m uvicorn app.main:app --app-dir "${BACKEND_DIR}" --reload --port "${PORT}"
else
    echo "Error: Unknown backend type in ${BACKEND_DIR}"
    exit 1
fi
