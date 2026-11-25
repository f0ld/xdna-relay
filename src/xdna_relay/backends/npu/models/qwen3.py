"""Qwen3 (dense) on IRON's shared decoder (``iron.lm``).

Qwen3 is Llama's layer with each q and k head RMS-normed before RoPE (``q_norm``,
``k_norm`` in the checkpoint; ``iron.lm.blocks.Attention(norms="qk")``), an explicit
``head_dim`` (128 at every size, so heads * head_dim != hidden_size on 0.6B), plain RoPE
(theta 1e6, no scaling) and a tied head on 0.6B/1.7B/4B, untied on 8B (``config.json``'s
``tie_word_embeddings``). Everything else is ``CausalLM``'s.

Projections default to ``Precision.MEDIUM`` (bfp16 macs): for prefill the KV it produces
decodes identically on Llama 3B (bench/results/2026-11-24-iron-prefill-profile.md); override
per executor (``IronLmExecutor(precision=...)``).

Profile: ``profiles/qwen3/<device>.json`` = IRON's Llama profile plus entries for Qwen3's
own shapes. Without the MHA entry for 32 heads x d=128 (8B), MHA runs at its defaults,
~10x slower (206 vs ~20 ms per layer-chunk; bench/results/2026-11-25-qwen3-iron.md).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from iron import Precision, lm
from iron.lm import CausalLM, DecoderConfig, DecoderLayer
from iron.lm.blocks import Attention


@dataclasses.dataclass(frozen=True)
class Qwen3Config(DecoderConfig):
    """``DecoderConfig`` and every projection's ``precision``."""

    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)


class Qwen3Layer(DecoderLayer):
    """Llama's layer, q and k heads RMS-normed (``q_norm``, ``k_norm``) before RoPE."""

    def __init__(self, c: Qwen3Config, i: int):
        super().__init__(c, i, self_attn=Attention(c, i, norms="qk", o_bias=c.attention_bias))


class Qwen3PackedLayer(DecoderLayer):
    """``Qwen3Layer`` whose prompt-chunk projections read pre-packed bfp16 weights (T-033)."""

    def __init__(self, c: Qwen3Config, i: int):
        from .packed import PackedAttention, PackedGatedMLP

        super().__init__(c, i, self_attn=PackedAttention(c, i, norms="qk", o_bias=c.attention_bias),
                         mlp=PackedGatedMLP(c))


class Qwen3(CausalLM):
    """Qwen3 on ``config``'s shape and a checkpoint's ``weights``."""

    config = Qwen3Config
    layer_class = Qwen3Layer
    profile = Path(__file__).with_name("profiles") / "qwen3"


class Qwen3Packed(Qwen3):
    """Qwen3 with packed bfp16 prompt-chunk projections; call ``packed.fill_packed`` after
    ``load_versions`` (``IronLmExecutor(packed=True)`` does).

    Two prompt-chunk versions share the arena (weights, packed states, KV cache): ``long``
    keeps each GEMM's packed B resident in the memtiles (read from DDR once per call, but the
    cores drain the replays of all 8 units even under a short row bound: ~0.25 s fixed on 8B),
    ``short`` streams it per 256-row unit (``k_tile=256``, profile ``qwen3-streamed``). The
    packed layout is the same for both (``pack_b`` does not depend on ``k_tile``).
    ``use_prompt_version(n)`` picks one per call (bench/results/2026-11-25-packed-gemm.md).
    """

    layer_class = Qwen3PackedLayer
    streamed_profile = Path(__file__).with_name("profiles") / "qwen3-streamed"
    short_max_rows = 640  # measured crossover (8B GEMMs: 2.5 units)

    def load_versions(self, *, release=None, full_elf=None, dev=None):
        from iron.common.image.packaging import require_device

        dev = dev or require_device()
        decode_shapes, prompt_shapes = self.shapes()
        decode = self.compile(dev=dev, full_elf=full_elf, **decode_shapes)
        feeds = decode if decode.emit is not None else None
        long = self.compile(dev=dev, feeds=feeds, full_elf=full_elf, **prompt_shapes)
        own = self.profile
        self.profile = self.streamed_profile
        try:
            short = self.compile(dev=dev, feeds=feeds, full_elf=full_elf, **prompt_shapes)
        finally:
            self.profile = own
        try:
            for version in (decode, long, short):
                version.load(release=release)
            self.loaded([decode, long])
        except BaseException:
            self.release()
            raise
        self.prompt_versions = {"long": long, "short": short}
        return self

    def use_prompt_version(self, new_rows: int) -> str:
        """Select the prompt-chunk version for a call evaluating ``new_rows`` tokens."""
        which = "short" if new_rows <= self.short_max_rows else "long"
        self.prompt_chunk = self.prompt_versions[which]
        return which


if __name__ == "__main__":
    lm.main(Qwen3)
