"""Qwen3 (dense) on IRON's shared decoder (``iron.lm``).

Qwen3 is Llama's layer with each q and k head RMS-normed before RoPE (``q_norm``,
``k_norm`` in the checkpoint; ``iron.lm.blocks.Attention(norms="qk")``), an explicit
``head_dim`` (128 at every size, so heads * head_dim != hidden_size on 0.6B), plain RoPE
(theta 1e6, no scaling) and a tied head on 0.6B/1.7B/4B, untied on 8B (``config.json``'s
``tie_word_embeddings``). Everything else is ``CausalLM``'s.

Projections default to ``Precision.MEDIUM`` (bfp16 macs): for prefill the KV it produces
decodes identically on Llama 3B (bench/results/2026-11-24-iron-prefill-profile.md); override
per executor (``IronLmExecutor(precision=...)``).

No tuned profile of its own: IRON's Llama profile directory is used, whose entries match
only the shapes Qwen3 shares with Llama; others take each operator's defaults.
"""
from __future__ import annotations

import dataclasses

from iron import Precision, lm
from iron.applications.llama.model import Llama
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


class Qwen3(CausalLM):
    """Qwen3 on ``config``'s shape and a checkpoint's ``weights``."""

    config = Qwen3Config
    layer_class = Qwen3Layer
    profile = Llama.profile


if __name__ == "__main__":
    lm.main(Qwen3)
