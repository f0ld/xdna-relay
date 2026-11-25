"""T-032: prefill placement policy (src/xdna_relay/sched/placement.py), pure logic.

Expected decisions follow the measured two-request benchmarks
(bench/results/2026-11-24-iron-prefill-profile.md, 2026-11-25-qwen3-iron.md): with a request
decoding, long prompts (~1.5k tokens) are cheaper on the NPU, short ones on the iGPU.
"""
import pytest

from xdna_relay.sched.placement import PlacementPolicy, costs_for, presets


@pytest.mark.parametrize("name", ["llama3.2-3b", "qwen3-8b"])
def test_long_prompt_with_decoder_goes_to_npu(name):
    p = PlacementPolicy(costs_for(name))
    assert p.choose(1494, active=1).device in ("npu", "split")
    assert p.choose(1536, active=1).device == "npu"  # whole blocks: nothing to split


@pytest.mark.parametrize("name", ["llama3.2-3b", "qwen3-8b"])
def test_short_prompt_stays_on_igpu(name):
    p = PlacementPolicy(costs_for(name))
    for active in (0, 1):
        assert p.choose(100, active=active).device == "igpu"


def test_npu_side_beyond_two_blocks_with_decoder():
    """Qwen3-8B with a request decoding: prompts past two NPU blocks always use the NPU (whole
    or split); tiny ones never do (measured 67 tokens: iGPU 0.57-0.72 s vs NPU 1.06 s)."""
    p = PlacementPolicy(costs_for("qwen3-8b"))
    for n in range(520, 2049, 8):
        assert p.choose(n, active=1).device in ("npu", "split"), n
    for n in range(8, 96, 8):
        assert p.choose(n, active=1).device == "igpu", n


def test_split_estimate_decomposes():
    """A split is the NPU's whole blocks (+ its wait) then the iGPU's tail (+ its wait)."""
    p = PlacementPolicy(costs_for("qwen3-8b"))
    c = p.costs
    s = p.estimate("split", 1702, active=1)
    assert s.npu_tokens == 1536
    npu = c.npu_alone_s(1536) * c.npu_slow + c.npu_wait_s
    tail = c.igpu_alone_s(166) * c.igpu_slow + c.igpu_wait_s
    assert s.ttft_s == pytest.approx(npu + tail)


@pytest.mark.parametrize("name,n,igpu,npu", [
    ("qwen3-8b", 67, 0.645, 1.06), ("qwen3-8b", 382, 1.83, 1.785), ("qwen3-8b", 1702, 5.81, 5.62),
    ("llama3.2-3b", 64, 0.32, 0.61), ("llama3.2-3b", 350, 0.725, 0.94), ("llama3.2-3b", 1494, 2.05, 2.345)])
def test_estimates_reproduce_measurements(name, n, igpu, npu):
    """Presets predict the measured B TTFT under A's decode within 15%
    (bench/results/2026-11-25-placement.md)."""
    p = PlacementPolicy(costs_for(name))
    assert p.estimate("igpu", n, active=1).ttft_s == pytest.approx(igpu, rel=0.15)
    assert p.estimate("npu", n, active=1).ttft_s == pytest.approx(npu, rel=0.15)


def test_decode_weight_zero_minimises_ttft_only():
    p = PlacementPolicy(costs_for("llama3.2-3b"), decode_weight=0.0)
    c = p.choose(1494, active=1)
    assert c.ttft_s <= c.other.ttft_s


def test_npu_context_limit_and_availability():
    p = PlacementPolicy(costs_for("qwen3-8b"), npu_max_tokens=2048)
    assert p.choose(3000, active=1).device == "igpu"
    p.npu_available = False
    assert p.choose(1702, active=1).device == "igpu"  # neither npu nor split


def test_igpu_prefix_hit_keeps_request_on_igpu():
    """A 1.7k-token prompt whose first 1.6k tokens the iGPU has cached: 100 new tokens there
    beat 1.7k on the NPU."""
    p = PlacementPolicy(costs_for("qwen3-8b"))
    assert p.choose(100, active=1, n_npu=1702, total=1702).device == "igpu"


def test_npu_queue_wait_counts():
    p = PlacementPolicy(costs_for("qwen3-8b"))
    assert p.choose(1702, active=1).device in ("npu", "split")
    assert p.choose(1702, active=1, npu_busy_s=5.0).device == "igpu"


def test_observe_moves_estimates():
    p = PlacementPolicy(costs_for("qwen3-8b"), ema=1.0)
    before = p.costs.npu_alone_s(1000)
    p.observe("npu", 1000, 2 * before, active=0)  # NPU twice as slow
    assert p.costs.npu_alone_s(1000) == pytest.approx(2 * before)
    tps = p.costs.igpu_prefill_tps
    p.observe("igpu", 1000, 1000 / (tps * 2), active=0)  # iGPU twice as fast
    assert p.costs.igpu_prefill_tps == pytest.approx(2 * tps)


def test_presets_are_copies():
    a = costs_for(presets()[0]); a.npu_block_s = 1.0
    assert costs_for(presets()[0]).npu_block_s != 1.0


def test_npu_model_matches_warm_curve():
    """bench/npu_prefill_curve.py, Qwen3-8B (2026-11-25): within 15% from 64 to 2000 tokens."""
    c = costs_for("qwen3-8b")
    for n, t in [(64, .599), (256, .697), (384, 1.166), (768, 1.813), (1493, 3.939), (2000, 4.810)]:
        assert c.npu_alone_s(n) == pytest.approx(t, rel=0.15), n
