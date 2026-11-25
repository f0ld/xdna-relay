"""Sample the iGPU while a benchmark row runs: shader clock (pp_dpm_sclk) and other processes'
GPU engine time (DRM fdinfo). An unexplained slow row then shows whether clocks dropped or
another client (a browser) competed. T-041: plain 4-request rows intermittently ran at half
speed; Chrome at ~40% engine time cost only ~4%.

    with GpuWatch() as w:
        ...
    w.summary()  -> "sclk 2105/1769 MHz, others 41% (chrome)"
"""
import glob
import os
import re
import threading
import time

_SCLK = next(iter(glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk")), None)


def _sclk() -> int | None:
    if not _SCLK:
        return None
    try:
        for line in open(_SCLK):
            if "*" in line:
                return int(re.search(r"(\d+)Mhz", line).group(1))
    except (OSError, AttributeError):
        return None
    return None


def _engine_ns() -> dict[str, int]:
    """pid -> summed amdgpu engine ns over its DRM clients (other processes only)."""
    me, per = str(os.getpid()), {}
    for fd in glob.glob("/proc/[0-9]*/fdinfo/*"):
        pid = fd.split("/")[2]
        if pid == me:
            continue
        try:
            txt = open(fd).read()
        except OSError:
            continue
        if "drm-driver:\tamdgpu" not in txt:
            continue
        cid = re.search(r"drm-client-id:\s*(\d+)", txt)
        ns = sum(int(x) for x in re.findall(r"drm-engine-\w+:\s*(\d+) ns", txt))
        key = (pid, cid.group(1) if cid else fd)
        per[key] = max(per.get(key, 0), ns)
    out = {}
    for (pid, _), ns in per.items():
        out[pid] = out.get(pid, 0) + ns
    return out


class GpuWatch:
    def __init__(self, every: float = 0.5):
        self.every, self.clocks = every, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            c = _sclk()
            if c:
                self.clocks.append(c)

    def __enter__(self):
        self._ns0, self._t0 = _engine_ns(), time.perf_counter()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        ns1, dt = _engine_ns(), time.perf_counter() - self._t0
        self.others = {}
        for pid, ns in ns1.items():
            pct = 100 * (ns - self._ns0.get(pid, ns)) / (dt * 1e9)
            if pct >= 1:
                try:
                    name = open(f"/proc/{pid}/comm").read().strip()
                except OSError:
                    name = pid
                self.others[name] = self.others.get(name, 0) + pct
        return False

    def summary(self) -> str:
        c = self.clocks
        clk = f"sclk {sum(c) // len(c)}/{min(c)} MHz" if c else "sclk n/a"
        tot = sum(self.others.values())
        top = max(self.others, key=self.others.get) if self.others else "-"
        return f"{clk}, others {tot:.0f}% ({top})"
