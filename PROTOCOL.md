# PROTOCOL — handoff phrases and file formats

Single source of truth for how the Architect (Claude Code) and the Implementer (OpenCode +
local Qwen) hand work to each other through the Operator. Both `CLAUDE.md` and `AGENTS.md`
point here. Architect-owned (protected).

## 1. Phrase format

Every phrase is **one line**. It names a task, an iteration and a file. The receiver reads that
file first and nothing else until needed.

```
@<KEYWORD> <TASK-ID> i<N> -> <file>
```

- `TASK-ID` is `T-` plus three digits (`T-001`).
- `i<N>` is the iteration, starting at `i1`. An iteration ends with a handoff file.
- The sender prints the phrase as the **very last line** of its response, alone, inside a
  code block, and the Implementer also runs `scripts/notify.sh "<phrase>"`.

## 2. Implementer → Architect (via Operator)

| Phrase | Meaning |
|---|---|
| `@REVIEW T-XXX iN -> handoff/T-XXX-iN.md` | Iteration done, done-command passes (or sub-goal reached). Please review. |
| `@ESCALATE T-XXX iN -> handoff/T-XXX-iN.md` | Stuck after 3 attempts or hardware problem. Please debug. |
| `@DESIGN T-XXX iN -> handoff/T-XXX-iN.md` | A design decision or spec change is needed before continuing. |
| `@DONE T-XXX iN -> handoff/T-XXX-iN.md` | Task complete, all done-criteria met. Please final-review and pick next task. |

## 3. Architect → Implementer (via Operator)

| Phrase | Meaning |
|---|---|
| `@TASK T-XXX i1 -> tasks/T-XXX-<slug>.md` | Start a new task from this spec (fresh session recommended). |
| `@VERDICT APPROVED T-XXX iN -> review/T-XXX-iN.md` | Accepted. Continue with the next step listed in the review. |
| `@VERDICT CHANGES T-XXX iN -> review/T-XXX-iN.md` | Apply the listed changes, then hand off as iteration N+1. |
| `@VERDICT FIXED T-XXX iN -> review/T-XXX-iN.md` | Architect fixed the issue directly (commits listed). Pull and continue. |
| `@VERDICT REDESIGN T-XXX iN -> tasks/T-XXX-<slug>.md` | Spec was rewritten. Reread it completely, start again at iteration N+1. |

## 4. Operator → either

| Phrase | Meaning |
|---|---|
| `@RESUME` | Session restarted. Read `PLAN.md`, find the task marked IN PROGRESS, read its latest handoff/review file, continue. |
| `@STATUS` | Reply with ≤5 lines: current task, iteration, last done-command result, blocker. No work. |

## 5. Files

| File | Written by | Template |
|---|---|---|
| `tasks/T-XXX-<slug>.md` | Architect | `tasks/TEMPLATE.md` |
| `handoff/T-XXX-iN.md` | Implementer, end of every iteration (≤ ~40 lines) | `handoff/TEMPLATE.md` |
| `review/T-XXX-iN.md` | Architect, for every handoff it answers | `review/TEMPLATE.md` |

Iteration numbering: a review `review/T-XXX-iN.md` answers `handoff/T-XXX-iN.md`. After
`@VERDICT CHANGES` or `REDESIGN` the Implementer's next handoff is `i<N+1>`.

## 6. Status lines in PLAN.md

Each task line in `PLAN.md` carries one status: `TODO`, `IN PROGRESS`, `REVIEW`, `DONE`,
`BLOCKED`. The Implementer sets `IN PROGRESS` when starting and `REVIEW` when handing off;
only the Architect sets `DONE`.

## 7. Branches and commits

- One task = one branch `task/T-XXX-<slug>`, created from `main`.
- Commit messages start with the task ID: `T-001: port vector_scalar_mul`.
- The Architect merges a task branch into `main` after `@DONE` is approved.
- Protected paths (`tests/reference/`, `tests/kernels/`, `tests/conftest.py`, `tasks/`,
  `review/`, `PROTOCOL.md`, `docs/ARCHITECTURE.md`, `docs/decisions/`, `scripts/hooks/`,
  `scripts/test.sh`) can only be committed with `ARCHITECT=1` (enforced by
  `scripts/hooks/pre-commit`).
