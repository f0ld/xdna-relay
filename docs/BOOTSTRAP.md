# BOOTSTRAP — xdna-relay

> **For Claude Code:** This file is a one-time setup brief. Read it fully, then build the
> repository structure, rule files and templates described below. When done, delete nothing,
> but move this file to `docs/BOOTSTRAP.md` and finish with the phrase in §9.
>
> **For the human:** Put this file in an empty git repo, start Claude Code there and say:
> `Read BOOTSTRAP.md and execute it.`

---

## 1. Project goal

Build an open, Linux-first LLM inference path for the AMD Ryzen AI XDNA2 NPU, starting small:

1. Correct and fast quantized kernels on the NPU (Q4 GEMM first).
2. NPU-side **prefill** for a transformer model, with **decode on the iGPU** via llama.cpp (Vulkan).
3. Optionally: ports of new models onto existing FastFlowLM kernels (`flm-add` style).

Later goal: a full NPU inference engine (prefill **and** decode on the NPU, own server and
scheduler). Not in scope yet, but **the architecture must stay open for it** (§1a). The name
"relay" refers to the scheduler handing work between NPU and iGPU — hybrid now, NPU-only later
as an option.

Not a goal: matching FastFlowLM's kernel performance or model catalog.

## 1a. Design constraints for future expansion

Record each as an ADR in `docs/decisions/` during setup:

1. **Backend-agnostic kernel interface.** Every op (GEMM, attention, norm, RoPE, …) is defined
   once with a contract (shapes, dtypes, layout). Behind it: an NPU implementation and a
   reference implementation; an iGPU implementation must be able to slot in later.
2. **KV cache as its own component** with a documented memory layout. Today it is handed to
   llama.cpp for decode; later it stays on the NPU. Both paths use the same layout spec.
3. **Scheduler separate from model code.** The model describes the computation graph; a
   scheduler decides which device runs which phase (prefill/decode) and when to hand off.

## 1b. Prior art to study first

Read these before designing kernels; reuse or learn from them instead of reinventing:

- `amd/IRON` — AMD's operator/graph layer on mlir-aie (may already provide needed ops)
- `Xilinx/mlir-aie` — toolchain and programming examples
- `atassis/xdna-engine` — community inference engine for XDNA2 (Rust + hand-written AIE kernels)
- `mistrjirka/llama.cpp-xdna2` — llama.cpp + XDNA2 integration attempt (relevant to M5)
- FastFlowLM (open runtime + open Gemma 4 engine) — reference for structure and Q4NX format

## 2. Hardware and software baseline

- Framework Laptop 13, Ryzen AI 9 HX 370 (Strix Point, XDNA2 NPU, 8 columns), Radeon 890M iGPU, 128 GB RAM
- Fedora, Podman/Toolbx available
- NPU stack: in-kernel `amdxdna` driver, XRT + XDNA plugin, NPU firmware ≥ 1.1.0.0, memlock unlimited
- Toolchain: IRON (Python API) + mlir-aie + Peano (AIE C++ compiler)
- Reference runtime for comparison: FastFlowLM (`flm validate`, `flm bench`)
- Reference numerics: PyTorch on CPU (ROCm on the 890M optional)

**Claude Code must verify current versions, install paths and APIs from the actual installed
packages and upstream docs/examples — not from memory — and record them in
`docs/knowledge/`.** Anything unverified is marked `UNVERIFIED`.

## 3. Roles

| Role | Who | Responsibilities |
|---|---|---|
| **Architect / Reviewer** | Claude Code (Opus), token-limited | Architecture, specs, test design, kernel design, reviews, hard debugging |
| **Implementer** | OpenCode + local Qwen3.6-35B-A3B, offline | Implements well-specified tasks against existing tests, benchmarks, scripts |
| **Operator** | Michael | Passes handoff phrases between the two, runs hardware, approves direction |

Token rule for the Architect: read the handoff file and the diff of the stated commit range
first. Read further files only when needed.

## 4. Repository structure to create

```
.
├── CLAUDE.md                 # Rules for Claude Code (Architect)
├── AGENTS.md                 # Rules for OpenCode (Implementer) — OpenCode reads this
├── PLAN.md                   # Milestones + ordered task list with status
├── PROTOCOL.md               # Handoff phrases and file formats (§6, §7) — single source of truth
├── docs/
│   ├── ARCHITECTURE.md       # Design of runtime, data layout, kernel interfaces
│   ├── BOOTSTRAP.md          # This file, moved here after setup
│   ├── decisions/            # ADR-style: NNNN-title.md, one decision per file (0001–0003 from §1a)
│   └── knowledge/            # Verified notes: toolchain, XRT API, AIE tiles, DMA, FLM formats, pitfalls
├── tasks/
│   ├── TEMPLATE.md
│   └── T-XXX-<slug>.md       # One spec per task, written by Architect
├── handoff/
│   ├── TEMPLATE.md
│   └── T-XXX-iN.md           # Implementer report per iteration
├── review/
│   ├── TEMPLATE.md
│   └── T-XXX-iN.md           # Architect verdict per iteration
├── src/                      # Kernels (IRON/AIE), host runtime, conversion tools
├── tests/
│   ├── reference/            # PyTorch reference implementations — PROTECTED
│   └── kernels/              # Correctness tests per kernel — PROTECTED
├── bench/                    # Benchmark scripts + results/*.csv
├── scripts/
│   ├── env-check.sh          # Driver, firmware, XRT, memlock, toolchain versions
│   ├── test.sh               # Runs tests; arg = task ID or test path; prints PASS/FAIL summary
│   ├── bench.sh
│   ├── notify.sh             # Desktop notification (notify-send) + terminal banner
│   └── hooks/pre-commit      # Enforces protected paths
└── opencode.json             # OpenCode config for local model (verify format in OpenCode docs)
```

## 5. Rules

### 5.1 Shared rules (put in both CLAUDE.md and AGENTS.md)

- All state lives in files, not in chat history. Sessions can be restarted at any time.
- One task = one branch `task/T-XXX-<slug>`. One iteration = one or more commits, ending
  with a handoff file.
- Every task has a **done-command** in its spec (e.g. `scripts/test.sh T-003`). Done means
  that command passes, nothing else.
- Hardware hangs: stop, do not retry more than once, document exact command and `dmesg` tail.
- Never invent API names. If unsure, search `docs/knowledge/` and installed sources/examples;
  if still unsure, escalate.

### 5.2 Implementer rules (AGENTS.md)

- Only work on the task named in the last `@TASK` / `@VERDICT` phrase. Read its spec first,
  then only the files the spec lists.
- **Never edit** `tests/reference/`, `tests/kernels/`, `tasks/`, `review/`, `PROTOCOL.md`,
  `docs/ARCHITECTURE.md`, `docs/decisions/`. The pre-commit hook enforces this.
- **3-attempt rule:** if the done-command still fails after 3 distinct fix attempts, stop and
  escalate. Do not weaken tolerances, skip tests or mock hardware to make tests pass.
- **Stop and escalate** immediately for: hardware hang/driver error, a needed design decision,
  a spec that contradicts the code or itself, or a change outside the spec's file list.
- **End of every iteration (mandatory, no exceptions):**
  1. Commit all work.
  2. Write `handoff/T-XXX-iN.md` from the template.
  3. Update the task's status line in `PLAN.md`.
  4. Run `scripts/notify.sh "<phrase>"`.
  5. Print the handoff phrase (§6) as the **very last line** of the response, alone,
     inside a code block, so the operator can copy it.
- An iteration should be small: aim for one sub-goal of the spec, not the whole task if it is large.
- Keep context short: prefer reading specific line ranges, do not dump large files or logs —
  summarize logs and save full output under `bench/` or `/tmp` with a path in the handoff.

### 5.3 Architect rules (CLAUDE.md)

- Write task specs that a ~3B-active model can execute: exact file list, interfaces, numbered
  steps, done-command, tolerances, known pitfalls. Prefer several small tasks over one big one.
- Write the tests **before** the implementer starts the task.
- Every review ends with a `review/` file and a verdict phrase (§6) as the very last line,
  alone, inside a code block.
- Record lasting knowledge (verified API facts, pitfalls found in review) in `docs/knowledge/`
  so the implementer stops repeating mistakes.
- Record design choices as ADRs in `docs/decisions/`.

## 6. Handoff phrases (copy-paste protocol)

Every phrase is one line. It names a task, an iteration and a file. The receiver reads that
file first and nothing else until needed. Format:

```
@<KEYWORD> <TASK-ID> i<N> -> <file>
```

### Implementer → Architect (via operator)

| Phrase | Meaning |
|---|---|
| `@REVIEW T-XXX iN -> handoff/T-XXX-iN.md` | Iteration done, done-command passes (or sub-goal reached). Please review. |
| `@ESCALATE T-XXX iN -> handoff/T-XXX-iN.md` | Stuck after 3 attempts or hardware problem. Please debug. |
| `@DESIGN T-XXX iN -> handoff/T-XXX-iN.md` | A design decision or spec change is needed before continuing. |
| `@DONE T-XXX iN -> handoff/T-XXX-iN.md` | Task complete, all done-criteria met. Please final-review and pick next task. |

### Architect → Implementer (via operator)

| Phrase | Meaning |
|---|---|
| `@TASK T-XXX i1 -> tasks/T-XXX-<slug>.md` | Start a new task from this spec (fresh session recommended). |
| `@VERDICT APPROVED T-XXX iN -> review/T-XXX-iN.md` | Accepted. Continue with the next step listed in the review. |
| `@VERDICT CHANGES T-XXX iN -> review/T-XXX-iN.md` | Apply the listed changes, then hand off as iteration N+1. |
| `@VERDICT FIXED T-XXX iN -> review/T-XXX-iN.md` | Architect fixed the issue directly (commits listed). Pull and continue. |
| `@VERDICT REDESIGN T-XXX iN -> tasks/T-XXX-<slug>.md` | Spec was rewritten. Reread it completely, start again at iteration N+1. |

### Operator → either

| Phrase | Meaning |
|---|---|
| `@RESUME` | Session restarted. Read `PLAN.md`, find the task marked IN PROGRESS, read its latest handoff/review file, continue. |
| `@STATUS` | Reply with ≤5 lines: current task, iteration, last done-command result, blocker. No work. |

## 7. File templates

### tasks/TEMPLATE.md
```
# T-XXX — <title>
Status: TODO | IN PROGRESS | REVIEW | DONE
Milestone: M<n>
Depends on: T-YYY

## Goal
<one paragraph>

## Files you may change
- src/...

## Files to read first
- docs/knowledge/...

## Interface / contract
<function signatures, data layouts, shapes, dtypes>

## Steps
1. ...

## Done-command
scripts/test.sh T-XXX   # must print PASS

## Tolerances / perf target
<e.g. max abs err 1e-2 vs reference; perf target only if stated>

## Known pitfalls
- ...
```

### handoff/TEMPLATE.md  (keep under ~40 lines)
```
# Handoff T-XXX iN — REVIEW | ESCALATE | DESIGN | DONE
Branch: task/T-XXX-<slug>
Commits: <first>..<last>

## Done this iteration
- ...

## Done-command result
`scripts/test.sh T-XXX` → PASS | FAIL (<k>/<n> tests, short summary)

## Attempts (only for ESCALATE)
1. <hypothesis> → <change> → <result>
2. ...
3. ...

## Evidence
- Error (≤15 lines, exact):
- Full logs: <path>
- dmesg tail (if hardware):

## Question for the Architect
<one precise question, or "none">

## Proposed next step
...
```

### review/TEMPLATE.md
```
# Review T-XXX iN — APPROVED | CHANGES | FIXED | REDESIGN

## Assessment
<2–5 lines>

## Required changes (CHANGES only)
1. <file:line> — <what and why>

## Architect commits (FIXED only)
- <hash> <message>

## Knowledge added
- docs/knowledge/<file>.md — <what>

## Next step for the Implementer
...
```

## 8. Setup tasks for this Claude Code session

Do these in order. Keep the operator informed with short status lines.

1. `git init` check, create the structure from §4.
2. Write `CLAUDE.md`, `AGENTS.md` from §5 (AGENTS.md must be self-contained: the implementer
   never reads this bootstrap file). Write `PROTOCOL.md` from §6 and §7, and reference it from
   both rule files.
3. Write the scripts:
   - `env-check.sh`: kernel version, `amdxdna` loaded, `/dev/accel/accel0`, NPU firmware
     version, XRT present, `ulimit -l`, mlir-aie/IRON/Peano versions, `flm validate` if present.
   - `test.sh`: runs pytest for a task ID or path, prints a one-line `PASS`/`FAIL k/n` summary
     at the end.
   - `notify.sh`: `notify-send "xdna-relay" "$1"` (if available) plus a clearly visible
     terminal banner with the phrase.
   - `hooks/pre-commit`: reject commits touching protected paths (§5.2) unless
     `ARCHITECT=1` is set in the environment. Install via `git config core.hooksPath scripts/hooks`.
4. Write `opencode.json` for an offline local OpenAI-compatible endpoint (llama.cpp server,
   Qwen3.6-35B-A3B, ≥64K context, tool calling enabled). Verify the current OpenCode config
   format first; leave model URL/name as clearly marked placeholders if unknown.
5. Run `scripts/env-check.sh`. Record results in `docs/knowledge/environment.md`. List
   anything missing as a TODO for the operator instead of guessing fixes with sudo.
6. Study the prior art in §1b (summarize each in `docs/knowledge/prior-art.md`: what it does,
   license, what we can reuse). Then research and write `docs/knowledge/` starter notes
   (verified, with source links/paths):
   `toolchain.md` (IRON, mlir-aie, Peano, how to build and run an example),
   `xdna2-architecture.md` (tiles, memory tiles, DMA, columns on Strix Point),
   `flm-formats.md` (Q4NX layout, model packaging, what is open vs. closed),
   `pitfalls.md` (empty, to be filled during reviews).
7. Write `docs/ARCHITECTURE.md` (first draft, built around the three constraints in §1a,
   with one ADR each) and `PLAN.md` with milestones:
   - **M0** Environment validated, an upstream mlir-aie example builds and runs on the NPU
   - **M1** Q4 GEMM correct vs. PyTorch reference
   - **M2** Q4 GEMM fast (beat a naive baseline, report vs. FLM where comparable)
   - **M3** One transformer layer (prefill) correct
   - **M4** Full-model prefill on the NPU
   - **M5** KV-cache handoff to llama.cpp decode on the iGPU
   - **M6** Decode on the NPU (matrix-vector kernels, KV cache stays on NPU)
   - **M7** Own OpenAI-compatible server + scheduler with modes: NPU-only, hybrid (NPU prefill + iGPU decode)

   M6–M7 are later goals: plan them coarsely, do not write task specs for them yet.
8. Write the tests and the spec for the first implementer task (M0 or M1, whichever the
   environment allows), so the operator can immediately start OpenCode.

## 9. Finish

Move this file to `docs/BOOTSTRAP.md`, commit with message `bootstrap: project structure and protocol`,
summarize in ≤10 lines what was created and any operator TODOs, then end with the start phrase
for the implementer as the last line, alone, in a code block, e.g.:

```
@TASK T-001 i1 -> tasks/T-001-<slug>.md
```
