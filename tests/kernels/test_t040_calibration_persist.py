"""T-040: persisting the online placement calibration (sched/online.py to_state/from_state,
sched/calibration_store.py, Engine(calibration_path=...))."""
import json

import pytest

from xdna_relay.sched.calibration_store import CalibrationStore, power_source, setup_key
from xdna_relay.sched.online import OnlineCosts, from_state, to_state
from xdna_relay.sched.placement import PlacementPolicy, costs_for

PRESET = "qwen3-8b-packed-batched"
SIZES = (83, 279, 398, 968, 1718)


def _trained() -> PlacementPolicy:
    p = PlacementPolicy.calibrating(costs_for("llama3.2-3b"))
    o = p.online
    for _ in range(5):
        o.observe_step(0.065, prefill_chunk=False, npu_busy=False)
    for i in range(12):
        p.observe_part("igpu", 83 + i, 0.41, active=1)
        p.observe_part("igpu", 398, 1.36, active=1)
        p.observe_part("npu", 397, 1.32, active=1)
        p.observe_part("npu", 1717, 4.3, active=1)
        p.observe_part("igpu", 1, 0.2, active=1)
        o.observe_window("igpu", 1.36, 3, n=398)
        o.observe_window("npu", 1.55, 16, n=397)
        o.observe_window("split", 1.65, 13, n=398)
    return p


def _decisions(p):
    return [(p.choose(n, 1).device, round(p.estimate("npu", n, 1).ttft_s, 6),
             round(p.estimate("split", n, 1).lost_decode_s, 6)) for n in SIZES]


def test_state_roundtrip_reproduces_estimates():
    p = _trained()
    d = json.loads(json.dumps(to_state(p.online)))  # through JSON, as on disk
    q = PlacementPolicy.calibrating(costs_for("llama3.2-3b"))
    from_state(q.online, d)
    assert _decisions(q) == _decisions(p)
    assert q.online.seen("npu", 397) == p.online.seen("npu", 397)  # no re-exploration


def test_incompatible_state_is_refused_and_leaves_fits():
    p = _trained()
    d = to_state(p.online)
    q = PlacementPolicy.calibrating(costs_for("llama3.2-3b"))
    before = _decisions(q)
    with pytest.raises(ValueError):
        from_state(q.online, {**d, "version": 99})
    with pytest.raises(ValueError):
        from_state(q.online, {**d, "short_rows": 123})
    assert _decisions(q) == before


def test_store_atomic_save_load_and_corrupt_file(tmp_path):
    p = _trained()
    path = tmp_path / "cal" / "setup.json"
    st = CalibrationStore(p.online, path, every_s=0.0, log=lambda m: None)
    st.save()
    assert path.exists() and not list(path.parent.glob(".*tmp"))
    q = PlacementPolicy.calibrating(costs_for("llama3.2-3b"))
    st2 = CalibrationStore(q.online, path, log=lambda m: None)
    assert st2.load() and st2.source.startswith("restored")
    assert _decisions(q) == _decisions(p)
    path.write_text("{not json")
    r = PlacementPolicy.calibrating(costs_for("llama3.2-3b"))
    msgs = []
    assert not CalibrationStore(r.online, path, log=msgs.append).load()
    assert msgs and "ignoring" in msgs[0]


def test_maybe_save_only_with_new_observations(tmp_path):
    p = PlacementPolicy.calibrating(costs_for(PRESET))
    path = tmp_path / "c.json"
    st = CalibrationStore(p.online, path, every_s=0.0, log=lambda m: None)
    st.maybe_save()
    assert not path.exists()  # nothing learned yet
    p.observe_part("igpu", 400, 1.4, active=1)
    st.maybe_save()
    assert path.exists()


def test_setup_key_separates_power_and_engine():
    a = setup_key(gguf="Qwen3-8B-Q4_K_M", npu="Qwen3-8B", packed=True, engine="batched-256",
                  prior=PRESET, power="ac")
    b = setup_key(gguf="Qwen3-8B-Q4_K_M", npu="Qwen3-8B", packed=True, engine="batched-256",
                  prior=PRESET, power="battery")
    c = setup_key(gguf="Qwen3-8B-Q4_K_M", npu="Qwen3-8B", packed=True, engine="batched-128",
                  prior=PRESET, power="ac")
    assert len({a, b, c}) == 3 and "/" not in a and a.startswith("Qwen3-8B-Q4_K_M")
    assert power_source() in ("ac", "battery", "unknown")


def test_engine_saves_on_close_and_restores(igpu, tmp_path):
    from tests.reference.models import QWEN3_06B_Q8, require
    from xdna_relay.server.engine import Engine

    path = tmp_path / "engine.json"
    prompt = ("<|im_start|>user\nTell me about rivers.<|im_end|>\n<|im_start|>assistant\n"
              "<think>\n\n</think>\n\n")
    e = Engine(require(QWEN3_06B_Q8), slots=2, n_ctx=4096, calibration_path=str(path))
    try:
        for _ in range(2):
            e.complete(e.tokenize(prompt), 16)
        learned = e.policy.online.summary()
    finally:
        e.close()
    assert path.exists()
    e2 = Engine(require(QWEN3_06B_Q8), slots=2, n_ctx=4096, calibration_path=str(path))
    try:
        assert e2.calibration.source.startswith("restored")
        s2 = e2.policy.online.summary()
        assert s2["igpu"] == learned["igpu"] and s2["plain_step_s"] == learned["plain_step_s"]
    finally:
        e2.close()
