#!/usr/bin/env bash
# Usage: scripts/bench.sh <bench/script.py> [args]
# Runs a benchmark script in the project env. Each script appends rows to
# bench/results/<name>.csv (header: date,commit,task,kernel,shape,dtype,iters,median_us,gflops,notes).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
. scripts/env.sh
script="${1:?usage: bench.sh <bench/script.py> [args]}"; shift
export RELAY_COMMIT="$(git rev-parse --short HEAD)"
python "$script" "$@"
