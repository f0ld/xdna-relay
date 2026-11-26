# Qwen3.6-35B-A3B (qwen35moe) on IRON: op inventory and graph plan (T-045)

Status 2026-11-26. Sources: `~/.cache/xdna-relay/llama.cpp/src/models/qwen35moe.cpp` (graph),
`src/models/delta-net-base.cpp` (recurrence), GGUF metadata of
`Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf`, installed IRON
(`.venv/lib64/python3.14/site-packages/iron`), iGPU profile
`bench/results/2026-11-26-moe-profile.md`. Claims not checked against source or hardware are
marked UNVERIFIED.

## 1. Shape (GGUF `qwen35moe.*`)

| | |
|---|---|
| layers | 40 trunk (+1 MTP/nextn block); layer i is full attention iff (i+1) % 4 == 0 → 10 attention, 30 gated delta-net (GDN) |
| hidden | 2048, RMSNorm eps 1e-6, pre-norm `attn_norm` and `post_attention_norm` (before the FFN) |
| vocab | 248320; `token_embd` Q8_0 (515 MiB), `output` Q6_K (398 MiB), untied |
| full attention | 16 q heads, 2 kv heads, head_dim 256; `attn_q` 2048→8192 = per head [q(256) ‖ gate(256)]; q/k RMSNorm per head (256); IMRoPE on 64 of 256 dims, sections [11,11,10,0], theta 1e7 (text: all position components equal → plain NeoX-style partial RoPE, UNVERIFIED for the interleave order); out = attn · sigmoid(gate) → `attn_output` 4096→2048 |
| GDN | `attn_qkv` 2048→8192 = q (16×128) ‖ k (16×128) ‖ v (32×128); `attn_gate` (z) 2048→4096; `ssm_beta`, `ssm_alpha` 2048→32 (F32); causal depthwise conv1d k=4 over the 8192 qkv channels then SiLU; q, k L2-normed; 16 k-heads repeated to 32 v-heads; gate g = softplus(alpha + dt_bias) · ssm_a; beta = sigmoid(beta); delta rule per v-head with state 128×128 fp32; output RMSNorm(128)·SiLU(z); `ssm_out` 4096→2048 |
| MoE | router `ffn_gate_inp` 2048→256 (F32), softmax, top-8, weights renormalised; experts gate/up 2048→512 (Q4_K), down 512→2048 (Q5_K, some Q6_K), SiLU-GLU; shared expert 512 (Q8_0) times sigmoid(`ffn_gate_inp_shexp`·x) |
| MTP | one full-attention block with MoE FFN; input eh_proj([enorm(embed(x_p)) ‖ hnorm(h_{p-1})]) 4096→2048; shares the LM head |

Recurrence (llama.cpp `build_delta_net`, per v-head, per token; q scaled by 1/sqrt(128)):

    S = exp(g) * S                       # g < 0: decay
    d = beta * (v - S^T k)               # 128
    S = S + k d^T                        # rank-1 update
    o = S^T q

Per-sequence state handed off: 10 attention layers × K,V (2×256 per token each); 30 GDN layers ×
conv state (3 × 8192) + S (32 × 128 × 128 fp32 = 2 MiB). ~61 MiB recurrent + 20 KiB/token KV.

## 2. Where the time goes on the iGPU (llama.cpp Vulkan, profile 2026-11-26)

Decode (1 token, 47 ms of ops ≈ 19.5 tok/s): bandwidth-bound, matmuls at 66–80 GB/s.
Q8_0 projections 45%, LM head 11%, experts 20%, recurrent state copies (GET_ROWS/CPY of the
2 MiB S per layer) ~8%, GATED_DELTA_NET 4%.
Prefill (512 tokens, 1.04 s): experts 46% (MUL_MAT_ID at 1.7–2.4 TFLOPS: 16 tokens per
expert), dense projections 24% (6.5 TFLOPS), elementwise/norm 16%, GDN + its concat + conv 13%,
flash attention 1%. Ubatch sweep pp2048: 128 → 305, 256 → 394, 512 → 481, 1024 → 460,
2048 → 431 tok/s.

Contention (bench/results/2026-11-24-baseline.md, -25-qwen3-iron.md): with both engines busy
the package (~15 W in low-power/balanced) and DRAM are shared: iGPU decode 20.6 → 15.7 tok/s
while the NPU prefills; an NPU prefill under iGPU decode takes ~1.6× as long. So for one user
the NPU is a **prefill engine** (long prompts: agents, files), not a decode accelerator.

## 3. Coverage in IRON (installed package)

| op | IRON | status |
|---|---|---|
| RMSNorm (2048, per-head 128/256) | `iron.lm.RMSNorm`, `operators/rms_norm.py` | have |
| Linear, bfp16 (`Precision.MEDIUM`) and packed | `iron.lm.Linear`, our `models/packed.py` | have (8B: 1493 tok / 2.6 s) |
| q/k norm attention, GQA | `iron.lm.blocks.Attention(norms="qk")`, `operators/mha.py` | have; head_dim 256 cases exist in mha tests (Gemma: 4/1 and 4/2 heads); 16/2 at d=256 needs a profile entry (8B lesson: untuned MHA ~10× slower) |
| partial RoPE (64 of 256) | `operators/rope.py` | UNVERIFIED: partial rotary support |
| sigmoid output gate | `operators/sigmoid.py`, `elementwise_mul.py` | have (compose) |
| causal depthwise conv1d with carried state | `iron.lm.CausalConv1d`, `operators/depthwise_conv1d.py`, `Checkpoints` | have (Granite 4.0-H) |
| SiLU, softplus | `operators/silu.py`; softplus none | softplus: small new kernel or host (32 values/token) |
| L2 norm per head | none (`magnitude.py` is complex magnitude) | new small kernel, or RMSNorm with weight 1/sqrt(128)·sqrt(128) trick: l2(x) = rms(x)/sqrt(d) → RMSNorm then scale (eps differs slightly) |
| gated delta rule | none; `operators/selective_scan.py` (Mamba2 scan, fp32 state, `state_slices.py` slices a head's state 16 rows per call) | **new kernel**, modelled on SelectiveScan |
| MoE router + top-k | none (`gemma4`, `granite_hybrid` refuse experts) | router GEMM have; softmax/top-8 on host (256 values/token) |
| expert FFN (grouped GEMM over variable token counts) | none | **new operator** |
| LM head | Linear | prefill needs only the last row: run on the iGPU or host |

## 4. Graph plan (prefill only; decode stays on the iGPU with MTP)

Per layer, one IRON call sequence per prefill pass of `T` tokens (T ≥ 1024, see §5):

1. `attn_norm` → GDN or attention → residual → `post_attention_norm`.
2. GDN: one Linear 2048→(8192+4096+32+32) (qkv, z, beta, alpha fused rows) →
   conv1d+SiLU (Checkpoints) → L2 norm q,k → **GatedDeltaRule** (state in/out, fp32) →
   RMSNorm(128)·SiLU(z) → Linear 4096→2048.
3. Attention: Linear 2048→8192+512+512 → q/k norm → partial RoPE → MHA (16/2, d=256, causal,
   KV written for handoff) → ·sigmoid(gate) → Linear 4096→2048.
4. MoE: router Linear 2048→256 (bf16 is enough? UNVERIFIED: routing flips with precision —
   keep F32 on host if needed: 2048×256×T MACs is cheap on the CPU) → host top-8 → host
   builds per-expert row lists → **ExpertFFN** (gather rows, gate/up GEMM, SiLU-GLU, down
   GEMM, scale by router weight, scatter-add) over all 256 experts in one call → shared
   expert (Linear pair) · sigmoid gate → residual.
5. Final: output norm on the last row; hand the hidden states back for the LM head (iGPU) and
   the MTP block (iGPU, from the exported per-position hidden states).

Weights for the NPU: bfp16 copies (~9 bits/weight): experts 32.2 B params ≈ 36 GB, rest
≈ 2.4 GB. RAM 125 GB (88 GB available with the 21 GB GGUF loaded) → fits; load once, keep
resident (start-up cost: dequant Q4_K/Q5_K → bfp16 once, cache on disk, ~40 GB). A Q4 dequant
kernel on the AIE (halves DRAM traffic) is a later optimisation, not the first version.

## 5. Cost model and go/no-go gates

- Weight streaming per pass: every expert is touched once T ≳ 512 (T×8 assignments over 256
  experts); bfp16 experts 36 GB at ~53 GB/s NPU read (xdna2-architecture.md) ≈ 0.7 s per
  pass, independent of T. Compute ≈ 6 GFLOP/token. At the 8B's measured ~9 TFLOPS effective:
  T=2048 → 1.4 s compute vs 0.7 s weights → ~1500 tok/s if overlapped; T=512 → weights
  dominate → ~700 tok/s at best. iGPU today: 420–480 tok/s. NPU passes must be large (≥ 2048
  tokens); short prompts stay on the iGPU (placement policy, calibrated online).
- **Gate 1 (T-046)**: ExpertFFN microbench on the NPU at 8/16/32/64/128 rows per expert, one
  layer's 256 experts: measured time per layer × 40 must beat the iGPU's 476 ms per 512
  tokens (0.93 ms/token) by ≥ 1.5× at T=2048, else stop and keep the NPU for dense models.
- **Gate 2 (T-047)**: GatedDeltaRule on the NPU ≤ 0.3 ms/token for 30 layers (iGPU: GDN +
  conv + concat ≈ 0.27 ms/token).
- Everything else reuses kernels the 8B path already runs at speed.

## 6. Handoff (NPU prefill → iGPU decode)

The iGPU sequence is restored from a llama.cpp sequence-state blob (`rl_state_seq_set`):
`llama_memory_hybrid::state_write` = attention KV part (`llama_kv_cache` format, as our
`kv/llama_state.py` writes for the 8B, per-arch K layout) then the recurrent part
(`llama_memory_recurrent::state_write_data`: s_trans, n_layer, per layer r type/row
size/rows (conv state), then per layer s type/row size/rows (S)). With MTP the shim's blob
(T-044) appends the MTP context's sequence state and the pending hidden state; an NPU
handoff has no MTP KV → either a shim entry restoring the target part only (MTP drafts start
cold: −6% on a code prompt, T-044) or an iGPU MTP-layer pass over the exported hidden states
(1 layer, cheap). Start with the first.

## 7. Task list

- T-046 ExpertFFN operator (IRON): grouped bfp16 GEMM over per-expert row lists, SiLU-GLU,
  weighted scatter-add; test vs NumPy on random routing; microbench (gate 1).
- T-047 GatedDeltaRule operator (fp32 state, sliced like SelectiveScan); test vs a NumPy port
  of `build_delta_net` (recurrent form) and vs llama.cpp state after a prompt; microbench
  (gate 2).
- T-048 Qwen35Moe graph on `iron.lm` (attention layer with gate and partial RoPE, GDN layer,
  MoE layer); weight conversion GGUF → bfp16 cache; teacher-forced check of the final hidden
  states vs llama.cpp on 3 prompts.
- T-049 Hybrid handoff blob (KV + conv + S), target-only restore with MTP; T-042-style
  continuation test.
- T-050 Placement: NPU prefill for long prompts (≥ threshold from calibration), iGPU decode
  with MTP; bench single-request TTFT on 2k/8k/16k prompts and two-request scenarios.

iGPU-side items found by the profile (llama.cpp fork patches, independent of IRON):
- in-place GDN state (drop the GET_ROWS/CPY of S per layer per token: ~8% of decode);
- prefill: solo prompts use the full 512-token ubatch (done in T-044: +24% TTFT).
