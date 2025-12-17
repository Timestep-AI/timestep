#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${APP_ROOT}/backend"
VENV_DIR="${BACKEND_DIR}/.venv"
PORT="${PORT:-8000}"

cd "$BACKEND_DIR"

UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --directory "${BACKEND_DIR}"
exec "${VENV_DIR}/bin/python" -m uvicorn app.main:app --app-dir "${BACKEND_DIR}" --reload --port "${PORT}"

