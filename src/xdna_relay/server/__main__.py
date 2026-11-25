"""``python -m xdna_relay.server``: serve a GGUF on the iGPU, optionally with NPU prefill.

Example (Qwen3-8B, two slots, NPU prefill with packed weights, auto placement):

    python -m xdna_relay.server --gguf Qwen3-8B-Q4_K_M.gguf --hf ~/.cache/xdna-relay/hf/Qwen3-8B \\
        --npu --packed --preset qwen3-8b-packed --slots 2 --port 8090
"""
import argparse
import signal
import sys

from xdna_relay.sched.placement import PlacementPolicy, costs_for, presets

from .chat import ChatTemplate
from .engine import Engine
from .http import serve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m xdna_relay.server")
    ap.add_argument("--gguf", required=True, help="target model for the iGPU (llama.cpp)")
    ap.add_argument("--hf", help="the same model's Hugging Face directory: chat template; NPU weights")
    ap.add_argument("--chat-template", help="a Jinja chat template file (default: from --hf)")
    ap.add_argument("--draft", help="draft GGUF for speculative decoding (batched: one shared draft context)")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--mtp", action="store_true",
                    help="speculative decoding with the model's own MTP head (GGUF with nextn layers)")
    ap.add_argument("--no-batch", action="store_true",
                    help="one llama.cpp context per slot, unbatched decode")
    ap.add_argument("--ctx", type=int, default=8192, help="context per slot")
    ap.add_argument("--prefill-chunk", type=int, default=256,
                    help="batched mode: prompt tokens per mixed step (smaller: decoders stall less)")
    ap.add_argument("--npu", action="store_true", help="NPU prefill worker (IRON) + KV handoff")
    ap.add_argument("--packed", action="store_true", help="NPU: packed bfp16 projections (Qwen3)")
    ap.add_argument("--npu-max-seq-len", type=int, default=2048)
    ap.add_argument("--preset", choices=presets(), help="measured placement costs for this model pair")
    ap.add_argument("--placement", default="auto", choices=["auto", "igpu", "npu", "split"])
    ap.add_argument("--no-persist", action="store_true",
                    help="do not load/save the online calibration (default: per-setup file)")
    ap.add_argument("--calibration-file", help="calibration state file (default: "
                    "~/.cache/xdna-relay/calibration/<setup key>.json)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="keep the preset fixed (default: fit it online from measured prefills)")
    ap.add_argument("--cache-mib", type=int, default=4096, help="prefix cache budget (host RAM)")
    ap.add_argument("--no-think", action="store_true", help="Qwen3: enable_thinking=False by default")
    ap.add_argument("--name", help="model id reported by /v1/models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="tokens per request when the request gives no max_tokens (thinking: ~4096)")
    a = ap.parse_args(argv)

    defaults = {"enable_thinking": False} if a.no_think else {}
    if a.chat_template:
        tpl = ChatTemplate(open(a.chat_template).read(), defaults=defaults)
    elif a.hf:
        tpl = ChatTemplate.from_hf_dir(a.hf, defaults=defaults)
    else:
        tpl = None  # the GGUF's own (tokenizer.chat_template), once the engine has loaded it
    batched = not a.no_batch
    preset = a.preset
    if preset and batched and f"{preset}-batched" in presets():
        preset = f"{preset}-batched"  # measured on the batched engine
    if preset:
        costs = costs_for(preset)
        policy = PlacementPolicy(costs) if a.no_calibrate else PlacementPolicy.calibrating(costs)
    else:
        policy = None
    cal_path = None
    if not a.no_calibrate and not a.no_persist:  # with a preset, or the engine's default prior
        from pathlib import Path

        from xdna_relay.sched.calibration_store import DEFAULT_DIR, power_source, setup_key
        key = setup_key(gguf=Path(a.gguf).stem, npu=Path(a.hf).name if (a.npu and a.hf) else None,
                        packed=a.packed and a.npu, engine=(f"batched-{a.prefill_chunk}" + ("-spec" if a.draft else "") + ("-mtp" if a.mtp else "")) if batched
                        else "slots", prior=preset or "default", power=power_source())
        cal_path = a.calibration_file or str(DEFAULT_DIR / f"{key}.json")
    engine = Engine(a.gguf, slots=a.slots, n_ctx=a.ctx, draft_gguf=a.draft, hf_dir=a.hf, npu=a.npu,
                    packed=a.packed, npu_max_seq_len=a.npu_max_seq_len, policy=policy,
                    placement=a.placement, cache_bytes=a.cache_mib << 20,
                    batched=batched, prefill_chunk=a.prefill_chunk, calibration_path=cal_path, mtp=a.mtp)
    if tpl is None:
        source = engine.shared.chat_template if engine.shared is not None else None
        if not source:
            engine.close()
            ap.error("no chat template in the GGUF: give --hf or --chat-template")
        tpl = ChatTemplate(source, defaults=defaults)
    name = a.name or engine.model
    httpd = serve(engine, tpl, name, a.host, a.port, default_max_tokens=a.max_tokens)
    print(f"xdna-relay serving {name} on http://{a.host}:{a.port} ({a.slots} slots, "
          f"{'batched' if engine.batched else 'unbatched'}"
          f"{', NPU prefill' if a.npu else ''}{', draft ' + a.draft if a.draft else ''}"
          f"{', placement preset ' + preset if preset else ''})"
          + (f"\n  calibration: {engine.calibration.path} ({engine.calibration.source})"
             if engine.calibration else ""), flush=True)

    def stop(*_):
        httpd.shutdown()

    # A background job from a non-interactive shell inherits SIGINT ignored; restore Ctrl-C.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda *_: (httpd.server_close(), engine.close(), sys.exit(0)))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        engine.close()


if __name__ == "__main__":
    main()
