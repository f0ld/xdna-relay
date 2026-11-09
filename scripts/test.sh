#!/usr/bin/env bash
# Usage: scripts/test.sh <T-XXX | path> [extra pytest args]
#   scripts/test.sh T-001                 -> all tests/**/test_t001_*.py
#   scripts/test.sh tests/kernels/x.py    -> that path
# Last line of output is always: PASS <n>/<n>  or  FAIL <passed>/<total> (...)
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
. scripts/env.sh
arg="${1:?usage: test.sh <T-XXX|path> [pytest args]}"; shift
if [[ "$arg" =~ ^T-([0-9]{3})$ ]]; then
    id="t${BASH_REMATCH[1]}"
    mapfile -t targets < <(find tests -name "test_${id}_*.py" | sort)
    if [ ${#targets[@]} -eq 0 ]; then
        echo "FAIL 0/0 (no tests found for $arg: tests/**/test_${id}_*.py)"; exit 1
    fi
else
    targets=("$arg")
fi
mkdir -p /tmp/xdna-relay
log="/tmp/xdna-relay/test-${arg//\//_}.log"
report="/tmp/xdna-relay/test-${arg//\//_}.xml"
python -m pytest -q -rfE --junitxml="$report" "${targets[@]}" "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
summary=$(python - "$report" <<'PY'
import sys, xml.etree.ElementTree as ET
try:
    r = ET.parse(sys.argv[1]).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    n, f, e, sk = (int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    print(f"{n - f - e - sk} {n} {f} {e} {sk}")
except Exception:
    print("0 0 0 1 0")
PY
)
read -r ok total fails errs skips <<<"$summary"
echo "full log: $log"
# Skips count as failures: done means every test ran and passed.
if [ "$rc" -eq 0 ] && [ "$total" -gt 0 ] && [ "$skips" -eq 0 ]; then
    echo "PASS $ok/$total"
else
    echo "FAIL $ok/$total (failures=$fails errors=$errs skipped=$skips)"
    [ "$rc" -eq 0 ] && rc=1
fi
exit "$rc"
