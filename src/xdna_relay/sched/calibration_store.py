"""Persisting the online placement calibration across server restarts (T-040).

One JSON file per setup, under ``~/.cache/xdna-relay/calibration/``: the key names what the
fitted costs depend on — the iGPU model, the NPU model and its packing, the engine mode
(batched, prefill chunk), the preset used as prior, and the power source (AC vs battery
changed times by up to ~10%), so each keeps its own calibration. Writes are atomic
(temporary file + rename), at most every ``every_s`` seconds after new observations, and on
``close``. A file that does not fit (another format version or layout) is ignored with a
notice; the server then starts from the prior.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from .online import from_state, to_state

DEFAULT_DIR = Path.home() / ".cache" / "xdna-relay" / "calibration"


def power_source() -> str:
    """"ac", "battery" or "unknown" (sysfs power supplies)."""
    try:
        for p in Path("/sys/class/power_supply").iterdir():
            t = (p / "type").read_text().strip() if (p / "type").exists() else ""
            if t == "Mains" and (p / "online").exists():
                return "ac" if (p / "online").read_text().strip() == "1" else "battery"
    except OSError:
        pass
    return "unknown"


def setup_key(**parts) -> str:
    """A readable, filesystem-safe key for a setup (its parts' values in the given order, flags
    by name) with a short hash of all of them, e.g.
    ``Qwen3-8B-Q4_K_M__Qwen3-8B__packed__batched-256__qwen3-8b-packed-batched__ac__1a2b3c4d``."""
    raw = json.dumps(parts, sort_keys=True)
    # in the caller's order; a True flag by its name
    words = [k if v is True else str(v) for k, v in parts.items() if v not in (None, "", False)]
    slug = re.sub(r"[^A-Za-z0-9_.+-]+", "-", "__".join(words))[:160]
    return f"{slug}__{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


class CalibrationStore:
    def __init__(self, online, path: Path, every_s: float = 30.0, log=print):
        self.online, self.path, self.every_s, self.log = online, Path(path), every_s, log
        self._lock = threading.Lock()
        self._saved_at = 0.0
        self._saved_obs = self._observations()
        self.source = "prior"

    def _observations(self) -> int:
        o = self.online
        return (sum(f.n for f in o.igpu.values()) + sum(f.n for f in o.npu.values())
                + o._plain_step.n)

    def load(self) -> bool:
        """Restore the file if present and compatible. Returns whether it was restored."""
        if not self.path.exists():
            return False
        try:
            d = json.loads(self.path.read_text())
            from_state(self.online, d["state"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log(f"calibration: ignoring {self.path} ({e}); starting from the prior")
            return False
        self._saved_obs = self._observations()
        self.source = f"restored {self.path.name} (saved {d.get('saved', '?')})"
        self.log(f"calibration: {self.source}")
        return True

    def save(self) -> None:
        with self._lock:
            data = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "state": to_state(self.online)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, self.path)  # atomic: a reader sees the old or the new file
            self._saved_at = time.monotonic()
            self._saved_obs = self._observations()

    def maybe_save(self) -> None:
        """Save if there are new observations and ``every_s`` passed since the last save."""
        if self._observations() != self._saved_obs and time.monotonic() - self._saved_at >= self.every_s:
            try:
                self.save()
            except OSError as e:
                self.log(f"calibration: could not save {self.path}: {e}")
