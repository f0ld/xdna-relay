"""T-028: per-operator time of an IRON model's prompt chunk on the NPU.

Records the prompt-chunk graph (weights never read), times each distinct operator design
alone with IRON's own harness (``iron.tune.measure.measure``, dispatch cancelled), at the
per-call values of an ``--rows``-token prompt, and multiplies by how often the graph runs it.

Usage: python bench/iron_op_profile.py --hf HF_DIR [--rows 1494] [--model llama]
"""
import argparse
import collections
import time

import aie.utils as aie_utils
from iron.common.image.packaging import require_device
from iron.lm.causal import Checkpoint, weights_file
from iron.lm.generation import Sampler
from iron.operators.sample import Sample
from iron.tune.autotune import Shape, Timed, _identity
from iron.tune.measure import Cache, Timing, measure, pmode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--rows", type=int, default=1494)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--precision", choices=["highest", "high", "medium"], help="override the model's")
    ap.add_argument("--chunk", type=int, help="prefill_chunk override (rows per prompt-graph call)")
    ap.add_argument("--only", help="time only operators whose class name starts with this")
    a = ap.parse_args()
    if a.model == "llama":
        from iron.applications.llama.model import Llama as cls
    else:
        raise SystemExit(f"unknown model {a.model}")

    import dataclasses
    weights = weights_file(a.hf)
    config = dataclasses.replace(cls.default_config(weights), max_seq_len=a.max_seq_len)
    if a.precision:
        from iron import Precision
        config = dataclasses.replace(config, precision=Precision(a.precision))
    if a.chunk:
        config = dataclasses.replace(config, prefill_chunk=a.chunk)
    model = cls(config, Checkpoint(weights).tree("unread"))
    C = config.prefill_chunk
    _, prompt_shapes = model.shapes()
    rec = model.record(**prompt_shapes)
    n = a.rows
    if n > C:  # time the last chunk of an n-row prompt (attention over the most context)
        start = (n - 1) // C * C
        values = dict(position=n - 1, token=0, start=start, rows=n - start,
                      rope_start=model.rope_row(n, start))
    else:
        values = dict(position=n - 1, token=0, start=0, rows=n, rope_start=model.rope_row(n, 0))
    contents = {}
    samples = {s.op.k_max for s in rec.steps if isinstance(s.op, Sample)}
    if samples:
        _, draws = rec.states[id(model.draws)]
        contents[draws.name] = Sampler().rows(config.max_seq_len, max(samples))
    timed = Timed(rec, values, contents)

    dev = require_device()
    full_elf = dev.can_dispatch_full_elf
    print(f"model {a.hf.rstrip('/').split('/')[-1]}; chunk {C} rows, timed at {n}; "
          f"precision {getattr(config, 'precision', '?')}; NPU power mode {pmode()}; "
          f"core {aie_utils.DefaultNPURuntime.core_mhz} MHz; full_elf {full_elf}", flush=True)

    count = collections.Counter()
    first = {}
    for step in rec.steps:
        ident = _identity(step.op, dev)
        count[ident] += 1
        first.setdefault(ident, step.op)
    print(f"{len(rec.steps)} steps, {len(first)} distinct designs", flush=True)

    cache = Cache.here()
    rows = []
    t0 = time.perf_counter()
    for ident, op in first.items():
        if a.only and not type(op).__name__.startswith(a.only):
            continue
        try:
            m = measure([op], full_elf=full_elf, timing=Timing(a.rounds, a.calls, 1, 1.5),
                        values=timed.op_values(op), inputs=timed.op_inputs(op), cache=cache,
                        log=lambda s: None)[0]
            us = m.t_step_us
        except Exception as e:  # report and go on: one design failing should not hide the rest
            print(f"  failed: {Shape(op)}: {type(e).__name__}: {e}", flush=True)
            us = float("nan")
        rows.append((us * count[ident], us, count[ident], str(Shape(op)), op.resolved(dev)))
        print(f"  {us:10.1f} us x{count[ident]:3d}  {Shape(op)}", flush=True)
    total = sum(r[0] for r in rows if r[0] == r[0])
    print(f"\ntimed in {time.perf_counter() - t0:.0f} s; sum of steps {total / 1e3:.1f} ms "
          f"for {n} rows ({n / (total / 1e6):.0f} tok/s)\n")
    print("| share | ms | per call us | calls | operator |")
    print("|---|---|---|---|---|")
    for tot, us, k, name, _ in sorted(rows, key=lambda r: -(r[0] if r[0] == r[0] else 0)):
        print(f"| {100 * tot / total:4.1f}% | {tot / 1e3:.1f} | {us:.0f} | {k} | {name} |")


if __name__ == "__main__":
    main()
