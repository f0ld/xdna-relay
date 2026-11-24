# Speculative decoding — measured facts (2026-11-24)

Machine and builds: see `bench/results/2026-11-24-spec-igpu.md`. llama.cpp `6b0f75b` (contio's
Vulkan build), our patched clone at `~/.cache/xdna-relay/llama.cpp` (branch `xdna-relay`).

## llama.cpp server flags (verified in `common/arg.cpp` at 6b0f75b)
- `-md <draft.gguf>` only **loads** a draft model; drafting needs `--spec-type draft-simple`
  as well (default spec type is none). Without it `timings.draft_n` stays 0.
- Draft length: `--spec-draft-n-max N` (old `--draft-max` was removed).
- Other types: `draft-mtp`, `draft-eagle3`, `draft-dflash`, `draft-dspark`, `ngram-simple`,
  `ngram-mod`, `ngram-map-k`, `ngram-cache`; combinable as a comma list.
- Per-request stats: `timings.draft_n`, `timings.draft_n_accepted`; per-position acceptance
  in the server log line `statistics draft-simple: ... #acc rate/pos`.

## Hybrid (recurrent) models: rollback cost
- Qwen3.5 (`qwen35` arch) has Gated-DeltaNet recurrent layers. Partial draft rejection needs
  the recurrent state rolled back.
- llama.cpp keeps `n_rs_seq` recurrent snapshots per sequence for cheap rollback
  (`llama_context_params.n_rs_seq`, EXPERIMENTAL; supported archs in
  `src/llama-arch.cpp: llm_arch_supports_rs_rollback`, includes QWEN35).
- At 6b0f75b `common_params_speculative::need_n_rs_seq()` (`common/common.h`) enables it only
  for MTP/EAGLE3/DFlash/DSpark — **not for draft-simple**. Then the target context falls back
  to full checkpoints: each cycle saves ~70 MB of state, and every partial rejection restores
  it and **replays** the accepted tokens (`spec_is_replay`, `tools/server/server-context.cpp`)
  → about one extra target forward pass per cycle.
- `draft-simple` feeds the target's verify batch to the draft context too
  (`common_speculative_impl_draft_simple::process`), so a hybrid draft model needs snapshots
  for that batch; its own drafting is token-by-token (several batches), which snapshots
  cannot undo (`llama_memory_recurrent::seq_rm`: "bounded by the snapshots of the last
  ubatch") → it also needs its checkpoint. Giving only one of the two aborts with
  "failed to remove sequence".
- Fix: `patches/llama.cpp/0001-spec-draft-simple-recurrent-rollback.patch` (+5–11% vs stock).
- Greedy spec decoding is not bit-identical to plain decoding here: batched verify vs
  single-token float differences flip near-ties (seen with stock llama.cpp too).

## Verify cost on the iGPU (llama-bench, Vulkan, performance profile)
Forward time vs batch size — small batches are almost free, so verify is not the problem:

| batch | 1 | 2 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (hybrid) ms | 49.9 | 49.9 | 53.8 | 59.2 | 67.4 | 102 |
| Qwen3-4B (dense) ms | 37.3 | 37.2 | 41.4 | 46.0 | 55.1 | 88.3 |

## Draft cost ratio
Qwen3.5-0.8B Q8_0 decodes at 62 tok/s vs the 4B at 21 → each drafted token costs ~0.33 of a
target step on the same device. That ratio, not verify, bounds same-device speedup; moving
the draft to the NPU (concurrent) removes it from the iGPU's critical path at the price of
the bandwidth contention measured in T-019.
