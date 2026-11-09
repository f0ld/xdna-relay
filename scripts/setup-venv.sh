#!/usr/bin/env bash
# Creates .venv/ with the pinned NPU toolchain (requirements-npu.txt) and installs this
# repo's package in editable mode. Idempotent. No root needed.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
py="${PYTHON:-python3}"
"$py" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,14), sys.version' \
    || { echo "need Python 3.11-3.14 (mlir-aie wheels)"; exit 1; }
[ -d .venv ] || "$py" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-npu.txt
pip install -q -e .
echo "venv ready: $root/.venv"
