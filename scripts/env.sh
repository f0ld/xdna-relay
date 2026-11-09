# Source this (do not execute): `. scripts/env.sh`
# Sets up XRT and the project venv. Safe to source repeatedly.
_relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    # shellcheck disable=SC1091
    . /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
fi
if [ -f "$_relay_root/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$_relay_root/.venv/bin/activate"
fi
# JIT artifacts (xclbin/insts) cache; default upstream is ~/.npu/cache
export NPU_CACHE_HOME="${NPU_CACHE_HOME:-$HOME/.npu/cache}"
unset _relay_root
