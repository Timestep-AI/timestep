#!/usr/bin/env bash
set -euo pipefail

# Add Deno to PATH if installed in home directory
export PATH="$HOME/.deno/bin:$PATH"

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_TYPE="${1:-python}" # Default to python if no argument is provided

if [ "$BACKEND_TYPE" = "python" ]; then
    BACKEND_DIR="${APP_ROOT}/backend"
    VENV_DIR="${BACKEND_DIR}/.venv"
    PORT="${PORT:-8000}"

    cd "$BACKEND_DIR"

    UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --directory "${BACKEND_DIR}"
    # Force reinstall timestep package to ensure latest changes are picked up
    uv pip install --python "${VENV_DIR}/bin/python" --force-reinstall --no-deps -e /home/mjschock/Projects/Timestep-AI/timestep/python || true
    exec "${VENV_DIR}/bin/python" -m uvicorn app.main:app --app-dir "${BACKEND_DIR}" --reload --port "${PORT}"
elif [ "$BACKEND_TYPE" = "backend-ts" ]; then
    BACKEND_DIR="${APP_ROOT}/backend-ts"
    PORT="${PORT:-8000}"

    cd "$BACKEND_DIR"

    exec deno run --allow-net --allow-env --allow-read --unstable-sloppy-imports --watch main.ts
else
    echo "Unknown backend type: $BACKEND_TYPE"
    exit 1
fi
