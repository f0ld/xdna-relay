"""Model files the hardware tests use (local paths on the Operator's machine).

Tests fail, never skip, when a file is missing (`require`). Download commands are in
bench/results/2026-11-24-t025-draft-placement.md and docs/knowledge/environment.md.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "build" / "native" / "libllama_shim.so"

QWEN3_8B_Q4 = Path.home() / ".local/share/contio-spike/gguf/Qwen3-8B-Q4_K_M.gguf"
QWEN3_06B_Q8 = Path.home() / ".cache/xdna-relay/gguf/Qwen3-0.6B-Q8_0.gguf"


def require(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"model file missing: {path}")
    return str(path)


def qwen3_chat(text: str) -> str:
    """Qwen3 chat prompt, one user turn, thinking disabled (empty think block)."""
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

LLAMA32_1B_HF = Path.home() / ".cache/xdna-relay/hf/Llama-3.2-1B-Instruct"
LLAMA32_1B_BF16 = Path.home() / ".cache/xdna-relay/gguf/Llama-3.2-1B-Instruct-BF16.gguf"
LLAMA32_1B_Q4 = Path.home() / ".local/share/contio-spike/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
QWEN3_06B_HF = Path.home() / ".cache/xdna-relay/hf/Qwen3-0.6B"
QWEN3_8B_HF = Path.home() / ".cache/xdna-relay/hf/Qwen3-8B"
QWEN36_35B_A3B_Q4 = Path.home() / ".local/share/contio-spike/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
QWEN36_35B_A3B_MTP_Q4 = Path.home() / ".local/share/contio-spike/gguf/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"
