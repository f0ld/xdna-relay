# 0002 — KV cache is its own component with a documented layout

Status: Accepted (2026-11-09)

## Context
In hybrid mode the NPU computes the KV cache during prefill and llama.cpp (iGPU) consumes it
for decode. In NPU-only mode it stays on the NPU. If the layout is an accident of whichever
kernel writes it, every mode change becomes a rewrite.

## Decision
The KV cache is a component (`src/xdna_relay/kv/`) with a written layout spec
(`docs/knowledge/kv-cache-layout.md`, drafted in M3): logical shape per layer
`[n_kv_heads, n_ctx, head_dim]`, dtype, physical block granularity, valid-range rules, and
`allocate / append / view / export / import` operations. Attention kernels read and write
through this layout; exporters convert it to llama.cpp's KV tensors.

## Consequences
- Both hybrid and NPU-only modes share one spec; the spec is versioned.
- The spec must be checked against llama.cpp's KV layout for the target model before M5.
- A layout conversion on handoff is allowed (measured), but it is in the exporter, not the kernel.
