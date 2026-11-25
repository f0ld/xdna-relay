"""Report NPU command timeouts (amdxdna TDR) that happened while a bench ran.

A TDR during a run can leave a result plausible but wrong, and IRON's harness does not
always raise (2026-11-25: three TDRs, no script error). ``guard()`` at the start of a bench
registers an exit hook that prints any ``aie2_tdr_detect`` kernel lines since then.
"""
import atexit
import datetime
import subprocess
import sys


def tdr_lines(since: str) -> list[str]:
    try:
        out = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-o", "short-iso"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # no journal access: say so rather than report "none"
        return [f"(cannot read kernel log: {e})"]
    return [l for l in out.splitlines() if "aie2_tdr_detect" in l]


def guard() -> None:
    since = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def report():
        lines = tdr_lines(since)
        if lines:
            print(f"WARNING: {len(lines)} NPU TDR event(s) during this run; results suspect:",
                  file=sys.stderr, flush=True)
            for l in lines:
                print("  " + l, file=sys.stderr, flush=True)
        else:
            print("tdr-guard: no NPU timeouts during this run", file=sys.stderr, flush=True)

    atexit.register(report)
