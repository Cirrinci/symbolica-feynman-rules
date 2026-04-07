#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
	CANDIDATE_PYTHON="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
	CANDIDATE_PYTHON="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
	CANDIDATE_PYTHON="python3.11"
elif command -v python3 >/dev/null 2>&1; then
	CANDIDATE_PYTHON="python3"
else
	echo "Error: no suitable Python 3 interpreter found on PATH."
	exit 1
fi

echo "Using interpreter: $CANDIDATE_PYTHON"
"$CANDIDATE_PYTHON" -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

# Fail early if Symbolica imports are not available in the created environment.
.venv/bin/python -c "from symbolica import S, Expression; from symbolica.community.spenso import Representation; from symbolica.community.idenso import simplify_metrics; print('Symbolica import check: OK')"

echo "Environment ready."
echo "Use it with: source .venv/bin/activate"
echo "Run the regression script with: .venv/bin/python src/examples.py --suite all --no-demo"
echo "Or use pytest with: .venv/bin/python -m pytest -q"
