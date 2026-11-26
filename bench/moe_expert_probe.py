"""T-046 step 1: what do Qwen3.6-35B-A3B's expert GEMMs cost on the NPU as plain IRON steps?

One full ELF holding E experts' FFN GEMMs back to back, each expert its own buffers (its own
packed bfp16 weights, so DDR traffic is real): gate+up ``ProjectionGEMM`` (K=2048, N=1024,
SiLU not applied: timing only) then down (K=512, N=2048), each with a per-call row bound
(``valid``) of the tokens routed to that expert. M=256: one 256-row unit (M_TILE 64 x 4 core
rows) per expert, enough for T <= ~8k prompt tokens at 8 of 256 experts.

Reports wall time per call of the whole runlist (median of --runs), per expert, and the
40-layer estimate at the given rows. The iGPU's experts take 0.93 ms per prompt token
(bench/results/2026-11-26-moe-profile.md): T=2048 -> 1.9 s for 40 layers.

Usage: python bench/moe_expert_probe.py --experts 1,16,64 --rows 64
"""
import argparse
import statistics
import time

from iron import Precision
from iron.operators.flm.gemm.design import Epilogue
from iron.operators.flm.gemm.op import ProjectionGEMM
from iron.tune.measure import Standalone, pmode

H, F = 2048, 512  # hidden, expert FFN width
# units padded to a multiple of 6: even (2 core rows) and a multiple of the 3 B slots, so the
# memtile ring keeps 3 slots (418 units: ring 2, 836 passes, NPU timeout 2026-11-26)
PAD_UNITS = 6


from xdna_relay.backends.npu.grouped_gemm import patch_standalone_bfp

patch_standalone_bfp()


def gemm(k, n, m=256):
    return ProjectionGEMM(M=m, K=k, N=n, epilogue_modes=(Epilogue.NONE,), precision=Precision.MEDIUM,
                          b_col_maj=False, bfp16_b=True)


def run(experts: int, rows: int, runs: int, kinds: str, m: int = 256):
    g, d = gemm(H, 2 * F, m), gemm(F, H, m)
    steps = []
    for _ in range(experts):
        if "g" in kinds:
            steps.append(g)
        if "d" in kinds:
            steps.append(d)
    t0 = time.perf_counter()
    sa = Standalone(f"moe_{kinds}_{experts}", steps, values={g: {"valid": rows}, d: {"valid": rows}})
    build = time.perf_counter() - t0
    sa.callable()  # warm
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sa.callable()
        ts.append(time.perf_counter() - t)
    return build, statistics.median(ts), min(ts)


def run_grouped(groups: int, runs: int, kinds: str, tile_n=None, core_rows=4):
    from xdna_relay.backends.npu.grouped_gemm import GroupedGEMM
    kw = ({} if tile_n is None else {"tile_n": tile_n}) | {"core_rows": core_rows}
    steps = []
    if "g" in kinds:
        steps.append(GroupedGEMM.of(groups, H, 2 * F, **kw))
    if "d" in kinds:
        steps.append(GroupedGEMM.of(groups, F, H, **kw))
    t0 = time.perf_counter()
    sa = Standalone(f"moe_grouped_{kinds}_{groups}_{tile_n}_{core_rows}", steps)
    build = time.perf_counter() - t0
    sa.callable()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sa.callable()
        ts.append(time.perf_counter() - t)
    return build, statistics.median(ts), min(ts)


def table_from_counts(path, T, layer, unit=128):
    """Unit -> expert table of one layer's routing: ceil(count / unit) units per expert, even."""
    import csv
    for r in csv.reader(open(path)):
        if int(r[0]) == T and int(r[1]) == layer:
            counts = list(map(int, r[2:]))
            table = [e for e, c in enumerate(counts) for _ in range(-(-c // unit))]
            pad = -len(table) % PAD_UNITS
            return table + table[-1:] * pad
    raise SystemExit(f"no row T={T} layer={layer} in {path}")


def run_table(table, runs, kinds, tile_n, core_rows):
    from xdna_relay.backends.npu.grouped_gemm import GroupedGEMM
    kw = {"core_rows": core_rows, "unit_groups": table} | ({} if tile_n is None else {"tile_n": tile_n})
    steps = []
    for kind, (k, n) in (("g", (H, 2 * F)), ("d", (F, H))):
        if kind not in kinds:
            continue
        m = GroupedGEMM.max_units(k)
        parts = -(-len(table) // m)  # calls of at most m units, as even as multiples of 6 allow
        size = -(-len(table) // parts)
        size += -size % 6
        for i in range(0, len(table), size):
            part = table[i:i + size]
            part = part + part[-1:] * (-len(part) % 6)
            steps.append(GroupedGEMM.of(256, k, n, **(kw | {"unit_groups": part})))
    t0 = time.perf_counter()
    sa = Standalone(f"moe_table_{kinds}_{len(table)}_{abs(hash(tuple(table))) % 10**8}_{tile_n}_{core_rows}", steps)
    build = time.perf_counter() - t0
    sa.callable()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sa.callable()
        ts.append(time.perf_counter() - t)
    return build, statistics.median(ts), min(ts)


def main():
    from tdr_guard import guard
    guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="1,16,64")
    ap.add_argument("--rows", default="64")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--m", type=int, default=256, help="static rows (256-row units) per expert GEMM")
    ap.add_argument("--grouped", action="store_true", help="one GroupedGEMM call per kind")
    ap.add_argument("--tile-n", type=int)
    ap.add_argument("--core-rows", type=int, default=4)
    ap.add_argument("--table", help="CSV of per-layer expert counts (bench/native/moe_routing MOE_DUMP)")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--kinds", default="gd", help="g: gate+up, d: down, gd: both")
    a = ap.parse_args()
    print(f"# NPU power mode {pmode()}", flush=True)
    print("kinds,experts,rows,build_s,median_ms,min_ms,per_expert_us,est_40_layers_s", flush=True)
    if a.table:
        from xdna_relay.backends.npu.grouped_gemm import GroupedGEMM
        unit = 64 * a.core_rows
        table = table_from_counts(a.table, a.T, a.layer, unit)
        build, med, mn = run_table(table, a.runs, a.kinds, a.tile_n, a.core_rows)
        print(f"# steps: {len(table)} units in calls of <= {[GroupedGEMM.max_units(k) for k in (H, F)]}", flush=True)
        print(f"table-{a.kinds}-tn{a.tile_n}-r{a.core_rows},T{a.T}/L{a.layer},{len(table)} units,{build:.1f},"
              f"{1e3 * med:.2f},{1e3 * mn:.2f},{1e6 * med / len(table):.1f},{med * 40:.2f}", flush=True)
        return
    if a.grouped:
        for e in map(int, a.experts.split(",")):
            build, med, mn = run_grouped(e, a.runs, a.kinds, a.tile_n, a.core_rows)
            print(f"grouped-{a.kinds}-tn{a.tile_n}-r{a.core_rows},{e},{64 * a.core_rows},{build:.1f},{1e3 * med:.2f},{1e3 * mn:.2f},"
                  f"{1e6 * med / e:.1f},{med / e * 256 * 40:.2f}", flush=True)
        return
    for rows in map(int, a.rows.split(",")):
        for e in map(int, a.experts.split(",")):
            build, med, mn = run(e, rows, a.runs, a.kinds, a.m)
            per = med / e
            print(f"{a.kinds},{e},{rows},{build:.1f},{1e3 * med:.2f},{1e3 * mn:.2f},{1e6 * per:.1f},"
                  f"{per * 256 * 40:.2f}", flush=True)


if __name__ == "__main__":
    main()
