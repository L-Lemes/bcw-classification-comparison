#!/usr/bin/env bash
# Create the virtual environment and install dependencies.
# Usage: ./setup.sh   (macOS / Linux)
set -euo pipefail

cd "$(dirname "$0")"

# Pick a Python 3.15 interpreter on macOS to avoid the outdated system python3.
# PYTHON can still override this for another supported interpreter.
if [ -n "${PYTHON:-}" ]; then
  :
elif command -v python3.15 >/dev/null 2>&1; then
  PYTHON="$(command -v python3.15)"
elif [ -x /opt/homebrew/opt/python@3.15/bin/python3.15 ]; then
  PYTHON="/opt/homebrew/opt/python@3.15/bin/python3.15"
else
  echo "Python 3.15 not found. Install it with: brew install python@3.15" >&2
  exit 1
fi

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 15), "Python 3.15+ required"'

echo "Python: $("$PYTHON" --version) ($(uname -m))"

# Create or refresh the environment with the selected interpreter.
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
  echo "Created .venv"
elif ! .venv/bin/python -c 'import sys; assert sys.version_info >= (3, 15)' >/dev/null 2>&1; then
  "$PYTHON" -m venv --clear .venv
  echo "Recreated .venv with Python 3.15+"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Environment ready. To use it:"
echo "  source .venv/bin/activate"
echo "  python run_experiments.py"
