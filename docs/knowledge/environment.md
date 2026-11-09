# Environment (verified 2026-11-09)

Source: `scripts/env-check.sh` run on the Operator's machine, plus `xrt-smi examine`,
`rpm -qa`, `/sys/class/accel/accel0/device/fw_version`. Re-run `scripts/env-check.sh` after
any system update and update this file.

## Hardware

| Item | Value | Source |
|---|---|---|
| Machine | Framework Laptop 13 (AMD Ryzen AI 300 Series), BIOS INSYDE 04.02 | `xrt-smi examine` |
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 24 threads | `xrt-smi examine` |
| RAM | **64 GB** (64069 MB) — the bootstrap brief said 128 GB; XRT reports 64 GB | `xrt-smi examine` |
| NPU | `NPU Strix`, PCI `1022:17f0` rev 10 at `0000:c2:00.1`, arch `aie2p`, topology `6x8` | `xrt-smi`, `lspci -nn` |
| iGPU | Radeon 890M, PCI `1002:150e` at `c1:00.0` | `lspci -nn` |

## Software

| Item | Value | Notes |
|---|---|---|
| OS | Fedora Linux 44 Workstation, glibc 2.43 | |
| Kernel | 7.2.14-200.fc44.x86_64 | in-tree `amdxdna` (driver reports version 0.10 to FLM) |
| NPU firmware | 1.1.2.64 | `/lib/firmware/amdnpu/17f0_10/` (from `linux-firmware-20261019`) |
| XRT | 2.27.0 (`xrt-base`, `xrt-npu`, `xrt_plugin-amdxdna` RPMs), in `/opt/xilinx/xrt` | `pyxrt` for CPython 3.14 in `/opt/xilinx/xrt/python`; needs `source /opt/xilinx/xrt/setup.sh` |
| memlock | unlimited | |
| `/dev/accel/accel0` | `crw-rw-rw- root:render` | |
| `tdr_timeout_ms` | **unknown** (sysfs param not readable by user); no `/etc/modprobe.d/amdxdna-tdr.conf` | default ≈2000 ms → commands die after ~4 s. See Operator TODO. |
| System Python | 3.14.8 | mlir-aie wheels support 3.11–3.14 (IRON docs/installation.md) |
| Project venv | `.venv/` (from `scripts/setup-venv.sh`, pins in `requirements-npu.txt`) | |
| mlir-aie | 1.4.6.dev15+ge5a3fa1 (wheel, find-links `latest-wheels-4`) | = mlir-aie commit e5a3fa1 (2026-11-07) |
| Peano (llvm-aie) | 22.0.0.2026102001+4f5df13d, at `.venv/lib64/python3.14/site-packages/llvm-aie` | |
| IRON | 0.1.0 from `git+https://github.com/amd/IRON@3c47c7a` (2026-11-09) | its requirements.txt pins the two wheels above |
| torch | 2.15.0+cpu (reference numerics) | system Python also has torch 2.14.1+rocm7.2, not used |
| OpenCode | v2.2.6 (`~/.opencode/bin/opencode`) | config schema v2; see toolchain.md |
| FastFlowLM | v1.0.9 at `~/.cache/claude-scratch/npu/flm` (env: `~/.cache/claude-scratch/npu/env.sh`), `flm validate` passes | models in `~/.config/flm/models` and `$FLM_MODEL_PATH` |

## Local services that share the hardware

- `contiod` on `127.0.0.1:8080/v1` is an OpenAI-compatible router. It serves
  `qwen3.6-35b-a3b` (llama.cpp ROCm container `ghcr.io/ggml-org/llama.cpp:server-rocm-b12063`,
  ctx 130560, `--jinja`, MTP speculative decoding) — that is the Implementer's model.
  It **also** has a FastFlowLM NPU backend (`qwen3-4b-npu`, alias `embed-gemma`). If that
  model gets loaded, it holds the NPU and our kernels will fail or contend.
  `scripts/env-check.sh` warns `npu-in-use` when another process has `/dev/accel/accel0` open.
- The Implementer's model runs on the iGPU. iGPU decode experiments (M5) will contend with it.

## Not yet verified (run on hardware)

- An mlir-aie / IRON design compiling and running on this NPU from `.venv` — that is T-001 (M0).
  Running freshly cloned upstream test code during bootstrap was blocked by the session's
  sandbox policy, so the first NPU run happens in T-001.
- llama.cpp Vulkan build on the 890M (needed for M5): `UNVERIFIED`, nothing installed yet.

## Operator TODOs

1. **Set the driver command timeout** (sudo; needed before larger kernels, M2+):
   ```
   echo "options amdxdna tdr_timeout_ms=150000" | sudo tee /etc/modprobe.d/amdxdna-tdr.conf
   sudo modprobe -r amdxdna && sudo modprobe amdxdna   # or reboot
   ```
   Source: IRON `docs/installation.md` §2. Small kernels (T-001) run with the default.
2. Before NPU work, make sure contio has no NPU model loaded (`curl -s 127.0.0.1:8080/v1/models`
   → `"device":"npu"` entries must show `"loaded":false`).
3. Optional, for M5: a llama.cpp Vulkan build (Fedora `vulkan-tools`, `glslc` / `shaderc`).
