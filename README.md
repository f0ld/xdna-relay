# xdna-relay

Open, Linux-first LLM inference for the AMD Ryzen AI XDNA2 NPU: quantized kernels and
transformer **prefill on the NPU**, **decode on the iGPU** (llama.cpp Vulkan), with a path to
NPU-only inference later.

- Plan and status: [PLAN.md](PLAN.md)
- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), decisions in [docs/decisions/](docs/decisions/)
- Verified facts: [docs/knowledge/](docs/knowledge/)
- Workflow (Architect ↔ Implementer handoffs): [PROTOCOL.md](PROTOCOL.md), [CLAUDE.md](CLAUDE.md), [AGENTS.md](AGENTS.md)

Setup: `scripts/setup-venv.sh && scripts/env-check.sh && git config core.hooksPath scripts/hooks`
