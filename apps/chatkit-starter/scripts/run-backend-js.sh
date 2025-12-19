#!/usr/bin/env bash
set -euo pipefail

# Add Deno to PATH if installed in default location
export PATH="$HOME/.deno/bin:$PATH"

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${APP_ROOT}/backend-js"
PORT="${PORT:-8000}"

cd "$BACKEND_DIR"

exec deno run --allow-net --allow-env --allow-read --unstable-sloppy-imports --watch main.ts
