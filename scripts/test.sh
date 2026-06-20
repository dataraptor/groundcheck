#!/usr/bin/env bash
# Run the full NO-KEY test suite: core + api + eval + browser e2e + axe audit.
#
# Runs each layer's pytest suite with `-m "not api"` (so the live-key smokes skip),
# then the Playwright e2e + axe-core audit under e2e/. Exits non-zero if any suite
# fails. The e2e tests skip cleanly (not fail) when chromium or network is absent.
#
# Usage:
#   ./scripts/test.sh
#   NO_INSTALL=1 ./scripts/test.sh    # skip the editable install
#   SKIP_E2E=1   ./scripts/test.sh    # core + api + eval only (no browser)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "${NO_INSTALL:-0}" != "1" ]; then
  python -m pip install -e "./core[dev]" -e "./api[dev]" >/dev/null
fi

export GROUNDCHECK_LLM=mock

declare -a suites=("core/tests:core" "api/tests:api (incl. contract)" "eval/tests:eval")
if [ "${SKIP_E2E:-0}" != "1" ]; then
  suites+=("e2e:e2e + axe audit")
fi

failed=0
for entry in "${suites[@]}"; do
  path="${entry%%:*}"
  name="${entry#*:}"
  echo ""
  echo "=== pytest ${path}  (${name}) ==="
  python -m pytest "${path}" -m "not api"
  if [ $? -ne 0 ]; then failed=1; fi
done

echo ""
if [ "$failed" -ne 0 ]; then
  echo "FAIL — at least one no-key suite failed."
  exit 1
else
  echo "PASS — all no-key suites green."
  exit 0
fi
