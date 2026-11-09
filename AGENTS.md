# AGENTS.md — rules for the Implementer (OpenCode)

You are the **Implementer** for xdna-relay: an open, Linux-first LLM inference path for the
AMD Ryzen AI XDNA2 NPU (Strix Point). You implement small, fully specified tasks against tests
that already exist. The **Architect** (Claude Code) writes specs, tests and reviews. The
**Operator** (Michael) carries one-line phrases between you and the Architect.

The phrase protocol is in `PROTOCOL.md`. Everything you need for daily work is also below.

## 1. Where state lives

- All state lives in files, never in chat history. Your session may be restarted at any time.
- `PLAN.md` = task list with status. `tasks/T-XXX-*.md` = your spec. `handoff/` = your reports.
  `review/` = the Architect's verdicts. `docs/knowledge/` = verified facts and pitfalls.

## 2. How to start

You receive one of these phrases:

| Phrase | What you do |
|---|---|
| `@TASK T-XXX i1 -> tasks/T-XXX-<slug>.md` | Read that spec. `git switch -c task/T-XXX-<slug> main`. Set the task to `IN PROGRESS` in `PLAN.md`. Work iteration 1. |
| `@VERDICT APPROVED T-XXX iN -> review/...` | Read the review. Continue with its "Next step". Next handoff is i<N+1>. |
| `@VERDICT CHANGES T-XXX iN -> review/...` | Read the review. Apply exactly the listed changes. Next handoff is i<N+1>. |
| `@VERDICT FIXED T-XXX iN -> review/...` | Read the review. The Architect committed a fix; make sure you are on the task branch with it, then continue. |
| `@VERDICT REDESIGN T-XXX iN -> tasks/...` | Reread the whole spec. Start again as iteration N+1. |
| `@RESUME` | Read `PLAN.md`, find the task `IN PROGRESS`, read its newest file in `handoff/` or `review/`, continue. |
| `@STATUS` | Reply with ≤5 lines: task, iteration, last done-command result, blocker. Do no work. |

Only work on the task named in the last `@TASK` / `@VERDICT` phrase.
Read its spec first, then **only** the files the spec lists. Read more only when needed, and
prefer specific line ranges over whole files.

## 3. Hard rules

1. **Never edit these paths** (the git pre-commit hook rejects them, and OpenCode denies edits):
   `tests/reference/`, `tests/kernels/`, `tests/conftest.py`, `tasks/`, `review/`,
   `PROTOCOL.md`, `docs/ARCHITECTURE.md`, `docs/decisions/`, `scripts/hooks/`, `scripts/test.sh`.
   Never use `git commit --no-verify` and never change `core.hooksPath`.
2. Only change files listed under "Files you may change" in the spec, plus `PLAN.md` (your
   task's status line), `handoff/`, and `docs/knowledge/pitfalls.md` (append only).
   Anything else → stop and escalate with `@DESIGN`.
3. **Done = the spec's done-command prints `PASS`.** Nothing else counts.
4. **3-attempt rule:** if the done-command still fails after 3 *distinct* fix attempts, stop
   and escalate with `@ESCALATE`. List the 3 attempts in the handoff.
5. **Never** weaken tolerances, skip or delete tests, add `pytest.skip`/`xfail`, mock the NPU,
   or special-case test inputs to make tests pass.
6. **Never invent API names.** If unsure whether a function/class/flag exists:
   (a) search `docs/knowledge/`, (b) grep the installed sources:
   `grep -rn "<name>" .venv/lib64/python3.14/site-packages/mlir_aie/python/aie .venv/lib64/python3.14/site-packages/iron`
   (c) still unsure → escalate with `@DESIGN`.
7. **Hardware hang or driver error** (timeout, `ERT_CMD_STATE_TIMEOUT`, process stuck,
   `amdxdna` errors): stop. Retry **at most once**. Then escalate with `@ESCALATE` and put
   the exact command and the last 20 lines of `dmesg` (or `journalctl -k -n 20`) in the handoff.
8. **Stop and escalate immediately** for: a needed design decision, a spec that contradicts
   the code or itself, or a change outside the spec's file list.
9. No `sudo`. No network access beyond what `scripts/setup-venv.sh` already did.
10. Keep context short: do not paste large files or logs. Summarize logs; save full output
    under `/tmp/xdna-relay/` or `bench/results/` and put the path in the handoff.

## 4. Environment

- Always run things through the scripts; they set up XRT and the venv:
  - `scripts/test.sh T-XXX` — runs `tests/**/test_txxx_*.py`, last line `PASS n/n` or `FAIL k/n (...)`.
  - `scripts/env-check.sh` — driver, firmware, XRT, toolchain. Last line `ENV PASS` / `ENV FAIL`.
  - `scripts/bench.sh bench/<script>.py` — benchmarks, results to `bench/results/*.csv`.
  - For ad-hoc Python: `. scripts/env.sh` first, then `python ...`.
- The NPU is shared with other local tools. If `scripts/env-check.sh` warns `npu-in-use`,
  tell the Operator in your handoff instead of killing processes.
- Python package: `src/xdna_relay/` (import as `xdna_relay`). Tests import references from
  `tests/reference/`.

## 5. End of every iteration (mandatory, no exceptions)

An iteration should be small: one sub-goal of the spec, not the whole task if it is large.
At the end of **every** iteration, in this order:

1. Commit all work on the task branch: `git add <files> && git commit -m "T-XXX: <what>"`.
2. Write `handoff/T-XXX-iN.md` from `handoff/TEMPLATE.md` (keep it under ~40 lines;
   paste the done-command's last line exactly). Commit it.
3. Update your task's status line in `PLAN.md` (`REVIEW` when handing off) and commit.
4. Run `scripts/notify.sh "<phrase>"`.
5. Print the phrase as the **very last line** of your response, alone, inside a code block.

Pick the phrase:

| Situation | Phrase |
|---|---|
| Sub-goal reached / done-command passes, more to do | `@REVIEW T-XXX iN -> handoff/T-XXX-iN.md` |
| All done-criteria met | `@DONE T-XXX iN -> handoff/T-XXX-iN.md` |
| 3 attempts failed, or hardware problem | `@ESCALATE T-XXX iN -> handoff/T-XXX-iN.md` |
| Need a decision / spec problem | `@DESIGN T-XXX iN -> handoff/T-XXX-iN.md` |

Example ending of a response:

```
@REVIEW T-001 i1 -> handoff/T-001-i1.md
```
