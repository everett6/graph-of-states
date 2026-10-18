#!/usr/bin/env bash
# Pure-Python math libraries for the agent's <run> sandbox (sympy, mpmath), copied from the training venv and
# byte-compiled with the sandbox's own interpreter (/usr/bin/python3), since the sandbox mounts them read-only.
# Code-grading sandboxes do not get them (LiveCodeBench-style grading stays standard library only).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=sandbox_libs
SRC=$(.venv-training/bin/python -c "import sympy, os; print(os.path.dirname(os.path.dirname(sympy.__file__)))")
rm -rf "$DEST.tmp" && mkdir -p "$DEST.tmp"
for package in sympy mpmath; do cp -r "$SRC/$package" "$DEST.tmp/"; done
find "$DEST.tmp" -name __pycache__ -prune -exec rm -rf {} +
/usr/bin/python3 -m compileall -q -j 0 "$DEST.tmp" >/dev/null
rm -rf "$DEST" && mv "$DEST.tmp" "$DEST"
du -sh "$DEST"
