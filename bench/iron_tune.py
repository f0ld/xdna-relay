"""T-028: tune IRON's operator settings for a model's prompt chunk, into a project profile.

IRON's shipped Llama profile (``iron/applications/llama/profiles/npu2.json``) names settings
for Llama 3.2 1B's shapes only; other sizes run each operator's ``resolve()`` defaults. This
runs IRON's own tuner (``iron.tune.autotune.autotune``) on the prompt-chunk graph at
``--precision`` and writes ``profiles/iron/<model>/<device>.json`` (the shipped entries kept,
covering ones replaced). ``IronLmExecutor(profile_dir=...)`` loads it.

Usage: python bench/iron_tune.py --hf HF_DIR [--precision medium] [--rows 2048]
"""
import argparse
import dataclasses
import functools
import shutil
from pathlib import Path

import aie.utils as aie_utils
from iron.common.declare import Profile
from iron.common.image.packaging import require_device
from iron.lm.causal import Checkpoint, weights_file
from iron.lm.generation import Sampler
from iron.operators.sample import Sample
from iron.tune.autotune import Timed, autotune, kernel_cycles
from iron.tune.measure import Cache, Timing, pmode

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--precision", default="medium")
    ap.add_argument("--rows", type=int, default=2048, help="prompt rows the chunk is timed at")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--out", type=Path, help="profile directory (default profiles/iron/<model>)")
    a = ap.parse_args()
    if a.model == "llama":
        from iron.applications.llama.model import Llama as cls
    else:
        raise SystemExit(f"unknown model {a.model}")
    from iron import Precision

    weights = weights_file(a.hf)
    config = dataclasses.replace(cls.default_config(weights), max_seq_len=a.max_seq_len,
                                 precision=Precision(a.precision))
    out_dir = a.out or ROOT / "profiles" / "iron" / a.model
    dev = require_device()
    path = out_dir / f"{dev.name}.json"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(cls.profile) / f"{dev.name}.json", path)

    class Tuned(cls):
        profile = out_dir

    model = Tuned(config, Checkpoint(weights).tree("unread"))
    _, prompt_shapes = model.shapes()
    rec = model.record(**prompt_shapes)
    n = a.rows
    values = dict(position=n - 1, token=0, start=0, rows=n, rope_start=model.rope_row(n, 0))
    contents = {}
    ks = {s.op.k_max for s in rec.steps if isinstance(s.op, Sample)}
    if ks:
        _, draws = rec.states[id(model.draws)]
        contents[draws.name] = Sampler().rows(config.max_seq_len, max(ks))
    print(f"power mode {pmode()}; tuning {path}", flush=True)
    tuned = autotune([Timed(rec, values, contents)], dev, Profile.load(path),
                     full_elf=dev.can_dispatch_full_elf,
                     timing=Timing(a.rounds, a.calls, 1, 1.5), cache=Cache.here(),
                     cycles=kernel_cycles(dev), core_mhz=aie_utils.DefaultNPURuntime.core_mhz,
                     log=functools.partial(print, flush=True))
    tuned.save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
