# Toolchain: IRON, mlir-aie, Peano

Verified 2026-11-09 against the installed `.venv` (mlir-aie 1.4.6.dev15+ge5a3fa1,
llvm-aie 22.0.0.2026102001+4f5df13d, IRON 0.1.0 @ 3c47c7a) by importing modules and reading
their sources. Running a design on the NPU from this venv is **not yet verified** (→ T-001).

## Install

```
scripts/setup-venv.sh      # creates .venv from requirements-npu.txt, pip install -e .
. scripts/env.sh           # sources /opt/xilinx/xrt/setup.sh + activates .venv
```

- Wheels are not on PyPI. `requirements-npu.txt` uses `--find-links`:
  - mlir-aie: `https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4`
  - llvm-aie (Peano): `https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly`
- Python 3.11–3.14 (cp314 wheels exist and install).
- `pyxrt` is **not** a pip package: it comes from `/opt/xilinx/xrt/python`, put on
  `PYTHONPATH` by XRT's `setup.sh`. Without sourcing it: `ModuleNotFoundError: pyxrt`.
- Install footprint: ~1 min download/install on this machine.

### Moving the pins

The mlir-aie wheel is a nightly. Bump the three pins together, from IRON's
`requirements.txt` at the new IRON commit:
`mlir_aie==…`, `llvm-aie==…`, `iron @ git+https://github.com/amd/IRON@<commit>`.
Then rerun every done-command. Record the new versions in environment.md.

## Where things are (in `.venv/lib64/python3.14/site-packages/`)

| Thing | Path / symbol |
|---|---|
| `aie` Python package | `mlir_aie/python/aie/` |
| `aie.iron` API | `mlir_aie/python/aie/iron/` |
| Kernel factories | `aie.iron.kernels` (e.g. `kernels.scale`, `kernels.quant.q4nx_dequant`), sources in `mlir_aie/include/aie_kernels/` |
| Compiler driver | `.venv/bin/aiecc` (also `aie-opt`, `aie-translate`); override with `AIECC_PATH` |
| Peano | `llvm-aie/` (`aie.utils.config.peano_install_dir()`), C++ at `llvm-aie/bin/clang++` |
| IRON | `iron/` (`iron.operators`, `iron.operators.flm`, `iron.common`, `iron.lm`) |
| JIT cache | `$NPU_CACHE_HOME`, default `~/.npu/cache` (`aie/utils/compile/__init__.py`) |

Other env vars read by the compile path: `AIE_KERNEL_COMPILE_JOBS`, `AIE_KERNEL_PCH`,
`AIE_XCLBINUTIL`, `MLIR_AIE_KERNEL_SOURCES`.

## Programming model (aie.iron)

A design declares data movement and compute; the runtime sequence moves host buffers.

- `ObjectFifo` — typed ping-pong (depth 2 default) channel between tiles (shim ↔ memtile ↔ core).
- `Worker` — a core program; calls `ExternalFunction`s (C++ kernels compiled by Peano).
- `Runtime` / `sequence(...)` — host-side fills/drains (`fill`, `drain`).
- `@iron.jit` — compiles on first call with given `CompileTime[...]` params, caches the
  xclbin/insts, then runs. Arguments annotated `In` / `Out` are NPU tensors.
- `iron.tensor(np_array, dtype=..., device="npu")`, `iron.zeros(n, dtype=..., device="npu")`,
  `iron.full(shape, value, dtype=..., device="npu")`; read back with `.numpy()`.
- Device: if none is set, `aie.utils.get_current_device()` probes the XRT runtime
  (→ NPU2 on this machine). Explicit: `iron.set_current_device(...)`; class
  `aie.iron.device.NPU2`. Partial-array devices: `AIEDevice.npu2_1col` … `npu2_7col`.
- `aie.iron.algorithms.transform(func, tensor_ty, *params, tile_size=16, trace_size=0)` —
  builds a one-core in → kernel → out design around a kernel with signature
  `(in_tile, out_tile, *params, tile_size)`.

## Smoke example: vector_scalar_mul (upstream)

Upstream: `mlir-aie/programming_examples/basic/vector_scalar_mul/vector_scalar_mul.py`
(at the pinned commit e5a3fa1). Core of it:

```python
import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform

@iron.jit
def vector_scalar_mul(A: In, C: Out, F: In, *, in1_size: CompileTime[int],
                      int_bit_width: CompileTime[int] = 16,
                      vectorized: CompileTime[bool] = True,
                      trace_size: CompileTime[int] = 0,
                      use_chess: CompileTime[bool] = False):
    in1_dtype = np.int16 if int_bit_width == 16 else np.int32
    tensor_size = in1_size // np.dtype(in1_dtype).itemsize   # in1_size is BYTES
    tile_size = tensor_size // 4
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    scale = kernels.scale(tile_size=tile_size, dtype=in1_dtype,
                          vectorized=vectorized, use_chess=use_chess)
    return transform(scale, tensor_ty, scalar_ty, tile_size=tile_size, trace_size=trace_size)

# call: argument order is (A, C, F)
a_t = iron.tensor(a_np, dtype=np.int16, device="npu")
f_t = iron.full((1,), 3, dtype=np.int32, device="npu")
c_t = iron.zeros(a_np.size, dtype=np.int16, device="npu")
vector_scalar_mul(a_t, c_t, f_t, in1_size=a_np.nbytes, int_bit_width=16)
result = c_t.numpy()
```

Constraints from upstream `_validate`: `in1_size` (bytes) multiple of 128 and ≥ 1024;
factor is one int32. Upstream run line for Strix: `python vector_scalar_mul.py -d npu2`.

## IRON operators

- Install check upstream: `pytest iron/tests/operators/test_catalog.py -k AXPY` (in an IRON
  checkout; tests are not in the pip-installed package).
- IRON op classes: `iron.common.Operator` subclasses with `build()`, `reference()`,
  `tolerance()`; calling `op(*buffers)` compiles once, then runs.
- `iron.operators.flm.ProjectionGEMM(M=..., K=..., N=..., epilogue=..., precision=..., bfp16_b=True)`,
  `op(A, op.pack_b(B), C_out)` (from `iron/operators/flm/gemm/README.md`).
  `Dequant` lives in `iron/operators/flm/dequant/op.py`. Exact constructor args: read
  the source before speccing M1 tasks.

## OpenCode config (v2.2.6, verified with `opencode debug config` / `opencode models`)

- Project `opencode.json` overrides global `~/.config/opencode/opencode.json`. `AGENTS.md`
  in the project root is discovered automatically.
- v2 schema: `providers.<id>.{name, package, settings{baseURL, apiKey}, models.<id>{name,
  capabilities{tools, input, output}, limit{context, output}}}`, `model: "<provider>/<model>"`,
  `permissions: [{action, resource, effect}]` (last match wins; default `ask`).
- v1 keys (`provider`, `npm`, `options`, `tool_call`) are still accepted and migrated
  (`npm: X` → `package: "aisdk:X"`, `tool_call` → `capabilities.tools`).
- Whether `permissions` edit globs match absolute or repo-relative paths: `UNVERIFIED`
  (patterns use a leading `*` to match both). The git pre-commit hook is the real guard.
