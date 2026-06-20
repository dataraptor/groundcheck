#!/usr/bin/env bash
# Bring up the full GroundCheck stack (API + static app) in mock mode — no API key.
#
# Installs the engine + API editable, then serves groundcheck_api with
# GROUNDCHECK_LLM=mock so the money demo runs end-to-end with no key. The page is
# served same-origin under /app, so it fetches /check with no CORS.
#
# Usage:
#   ./scripts/dev.sh           # serve on http://127.0.0.1:8000/
#   ./scripts/dev.sh 8137      # serve on a different port
#   NO_INSTALL=1 ./scripts/dev.sh   # skip the pip install step
set -euo pipefail

PORT="${1:-8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "${NO_INSTALL:-0}" != "1" ]; then
  echo "Installing core + api (editable)..."
  python -m pip install -e "./core[dev]" -e "./api[dev]"
fi

export GROUNDCHECK_LLM=mock
echo ""
echo "GroundCheck is up at  http://127.0.0.1:${PORT}/   (mock mode, no key)"
echo "  POST /check   GET /examples   GET /health   GET /app/GroundCheck.dc.html"
echo "Press Ctrl+C to stop."
echo ""
exec python -m uvicorn groundcheck_api.main:app --host 127.0.0.1 --port "${PORT}"
