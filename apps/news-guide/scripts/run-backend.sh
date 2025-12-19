#!/usr/bin/env bash
set -euo pipefail

# Determine the backend directory to run (default: backend)
BACKEND_ARG="${1:-backend}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${APP_DIR}/${BACKEND_ARG}"
PORT="${PORT:-8000}"

if [[ ! -d "${BACKEND_DIR}" ]]; then
  echo "Error: Backend directory '${BACKEND_DIR}' not found."
  exit 1
fi

if [[ "${BACKEND_ARG}" == "backend-ts" ]]; then
  # TypeScript/Deno backend
  export PATH="$HOME/.deno/bin:$PATH"
  cd "${APP_DIR}/../.."
  exec deno run \
    --allow-net \
    --allow-env \
    --allow-read \
    --unstable-sloppy-imports \
    apps/news-guide/backend-ts/main.ts
else
  # Python backend (default)
  VENV_DIR="${BACKEND_DIR}/.venv"
  cd "$BACKEND_DIR"
  UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --directory "${BACKEND_DIR}"
  uv pip install --python "${VENV_DIR}/bin/python" --force-reinstall --no-deps -e /home/mjschock/Projects/Timestep-AI/timestep/python || true
  exec "${VENV_DIR}/bin/python" -m uvicorn app.main:app --app-dir "${BACKEND_DIR}" --reload --port "${PORT}"
fi
