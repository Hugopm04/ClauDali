#!/usr/bin/env bash
# Starts the ClauDali server (web UI + HTTP API) using the project's own venv.
#
# Usage: ./scripts/start.sh [--host 127.0.0.1] [--port 8188]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
  echo "ClauDali is not installed yet." >&2
  echo "Run:  python -m installer install" >&2
  exit 1
fi

exec "$python" -m claudali serve "$@"
