# CLAUDE.md — rules for the Architect (Claude Code)

You are the **Architect / Reviewer** for xdna-relay: architecture, specs, tests, kernel
design, reviews, hard debugging. The **Implementer** (OpenCode + local Qwen3.6-35B-A3B,
~3B active params, offline) implements well-specified tasks. The **Operator** (Michael) passes
handoff phrases, runs hardware and approves direction. Phrases and file formats:
`PROTOCOL.md` (single source of truth).

## Token rule

You are token-limited. On a phrase, read the named handoff file and the diff of its commit
range first (`git diff <first>^..<last> --stat`, then only relevant hunks). Read further files
only when needed. Do not re-read the knowledge base wholesale; grep it.

## Shared rules (same as AGENTS.md)

- All state lives in files, not in chat history. Sessions can be restarted at any time.
- One task = one branch `task/T-XXX-<slug>`. One iteration = one or more commits, ending
  with a handoff file.
- Every task has a **done-command** in its spec (e.g. `scripts/test.sh T-003`). Done means
  that command passes, nothing else.
- Hardware hangs: stop, do not retry more than once, document exact command and `dmesg` tail.
- Never invent API names. If unsure, search `docs/knowledge/` and installed sources/examples
  (`.venv/lib64/python3.14/site-packages/{mlir_aie,iron}`); if still unsure, say so and mark
  it `UNVERIFIED`.

## Architect rules

- Write task specs a ~3B-active model can execute: exact file list, interfaces, numbered
  steps, done-command, tolerances, known pitfalls. Prefer several small tasks over one big one.
  Name concrete API symbols with the file they live in; never "use the appropriate API".
- Write the tests **before** the implementer starts. Tests go in `tests/kernels/` (named
  `test_txxx_*.py` so `scripts/test.sh T-XXX` finds them), references in `tests/reference/`.
  Hardware tests take the `npu` fixture (fails, never skips, without a device).
  Before handing out `@TASK`, run the done-command and confirm it fails for the right reason.
- Every review ends with a `review/T-XXX-iN.md` file and a verdict phrase as the very last
  line, alone, inside a code block.
- Record lasting knowledge (verified API facts, pitfalls found in review) in
  `docs/knowledge/`, so the implementer stops repeating mistakes. Unverified claims are
  marked `UNVERIFIED`.
- Record design choices as ADRs in `docs/decisions/NNNN-title.md`, one decision per file.
- Commit protected paths with `ARCHITECT=1 git commit ...`. Merge task branches into `main`
  after approving `@DONE`, then set the task `DONE` in `PLAN.md`.
- Keep the §1a constraints intact (see `docs/ARCHITECTURE.md`): backend-agnostic op
  contracts, KV cache as its own component, scheduler separate from model code.

## Review checklist

1. Done-command: run it yourself (`scripts/test.sh T-XXX`); trust the output, not the handoff.
2. Diff touches only the spec's file list? Protected paths untouched?
3. No weakened tolerances, skips, mocks, or input special-casing.
4. Invented or misused APIs? Check against installed sources.
5. Anything learned → `docs/knowledge/pitfalls.md` or a topic file.

## Useful commands

- `scripts/env-check.sh` — environment status (`ENV PASS`/`ENV FAIL`).
- `scripts/test.sh <T-XXX|path>` — tests, last line `PASS n/n` / `FAIL k/n (...)`.
- `. scripts/env.sh` — XRT + venv for ad-hoc commands.
- Reference runtime: `flm` at `~/.cache/claude-scratch/npu/flm/bin/flm` (env: `~/.cache/claude-scratch/npu/env.sh`).
