#!/usr/bin/env bash
# Checks driver, firmware, XRT, memlock and toolchain. Read-only; never uses sudo.
# Exit 0 if every REQUIRED check passes. WARN lines are advisory.
# Usage: scripts/env-check.sh [--no-flm]
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
. scripts/env.sh

fails=0
ok()   { printf '  OK    %-22s %s\n' "$1" "$2"; }
warn() { printf '  WARN  %-22s %s\n' "$1" "$2"; }
bad()  { printf '  FAIL  %-22s %s\n' "$1" "$2"; fails=$((fails + 1)); }

echo "xdna-relay env-check ($(date -Iseconds))"

ok kernel "$(uname -r)"

if lsmod | grep -q '^amdxdna'; then ok amdxdna "module loaded"; else bad amdxdna "module not loaded"; fi

if [ -c /dev/accel/accel0 ]; then
    if [ -r /dev/accel/accel0 ] && [ -w /dev/accel/accel0 ]; then
        ok device "/dev/accel/accel0 (rw)"
    else
        bad device "/dev/accel/accel0 not rw for $(id -un) (render group?)"
    fi
else
    bad device "/dev/accel/accel0 missing"
fi

fw=$(cat /sys/class/accel/accel0/device/fw_version 2>/dev/null || echo "")
if [ -z "$fw" ]; then
    bad npu-firmware "cannot read /sys/class/accel/accel0/device/fw_version"
else
    # require >= 1.1.0.0
    if [ "$(printf '%s\n1.1.0.0\n' "$fw" | sort -V | head -1)" = "1.1.0.0" ]; then
        ok npu-firmware "$fw"
    else
        bad npu-firmware "$fw (< 1.1.0.0)"
    fi
fi

tdr=$(cat /sys/module/amdxdna/parameters/tdr_timeout_ms 2>/dev/null || echo "unreadable")
if [ -f /etc/modprobe.d/amdxdna-tdr.conf ]; then
    ok tdr_timeout_ms "$tdr; $(grep -h tdr /etc/modprobe.d/amdxdna-tdr.conf)"
else
    warn tdr_timeout_ms "$tdr; no /etc/modprobe.d/amdxdna-tdr.conf (IRON wants 150000 for long commands)"
fi

memlock=$(ulimit -l)
if [ "$memlock" = "unlimited" ]; then ok memlock "unlimited"; else bad memlock "$memlock KiB (need unlimited)"; fi

if [ -f /opt/xilinx/xrt/version.json ]; then
    xv=$(python3 -c 'import json;print(json.load(open("/opt/xilinx/xrt/version.json"))["BUILD_VERSION"])' 2>/dev/null)
    ok xrt "$xv (/opt/xilinx/xrt)"
else
    bad xrt "/opt/xilinx/xrt missing"
fi
if command -v xrt-smi >/dev/null 2>&1; then
    dev=$(xrt-smi examine 2>/dev/null | grep -E '^\|\[' | head -1 | tr -s ' ')
    [ -n "$dev" ] && ok xrt-smi "$dev" || bad xrt-smi "no NPU listed by xrt-smi examine"
else
    bad xrt-smi "not on PATH (source /opt/xilinx/xrt/setup.sh)"
fi

if [ -x .venv/bin/python ]; then
    ok venv "$(.venv/bin/python --version) at .venv"
    python - <<'PY' || fails=$((fails + 1))
import importlib.metadata as md, sys
bad = 0
for dist, label in [("mlir_aie", "mlir-aie"), ("llvm-aie", "peano(llvm-aie)"),
                    ("iron", "iron"), ("torch", "torch"), ("numpy", "numpy"),
                    ("ml_dtypes", "ml_dtypes"), ("pytest", "pytest")]:
    try:
        print(f"  OK    {label:<22} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  FAIL  {label:<22} not installed (scripts/setup-venv.sh)"); bad += 1
try:
    import pyxrt  # noqa: F401
    print(f"  OK    {'pyxrt':<22} importable")
except Exception as e:
    print(f"  FAIL  {'pyxrt':<22} {e}"); bad += 1
try:
    from aie.utils import config
    print(f"  OK    {'peano dir':<22} {config.peano_install_dir()}")
except Exception as e:
    print(f"  FAIL  {'peano dir':<22} {e}"); bad += 1
sys.exit(1 if bad else 0)
PY
else
    bad venv ".venv missing (run scripts/setup-venv.sh)"
fi

# Other NPU users: the NPU is effectively single-tenant for our purposes.
holders=$(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c 'accel0' || true)
if [ "${holders:-0}" -gt 0 ]; then
    warn npu-in-use "$holders open fd(s) on /dev/accel/accel0 by other processes (contio/flm?)"
else
    ok npu-in-use "no other process holds /dev/accel/accel0"
fi

if [ "${1:-}" != "--no-flm" ]; then
    flm_bin=$(command -v flm || ls "$HOME/.cache/claude-scratch/npu/flm/bin/flm" 2>/dev/null || true)
    if [ -n "$flm_bin" ]; then
        if out=$(timeout 120 "$flm_bin" validate 2>&1); then
            ok flm "$("$flm_bin" version 2>/dev/null | head -1) validate passed ($flm_bin)"
        else
            warn flm "validate failed: $(echo "$out" | tail -1)"
        fi
    else
        warn flm "not found (optional reference runtime)"
    fi
fi

echo
if [ "$fails" -eq 0 ]; then echo "ENV PASS"; else echo "ENV FAIL ($fails required check(s) failed)"; fi
[ "$fails" -eq 0 ]
