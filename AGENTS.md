# AGENTS.md

## Scope

This repository uses the `codex-harness` workflow adapted for Neo. Use this file as the repo-local contract for Codex and agent behavior.

If a more specific user instruction conflicts with this file, follow the user instruction for the current task and update this file only when the workflow contract itself changes.

## Core Rules

- Treat MLX as the frozen scientific reference backend.
- Align PyTorch toward MLX; do not change MLX runtime semantics to make PyTorch pass.
- Keep `use_checkpoint: false` as the trusted PyTorch parity path unless a later explicit task proves otherwise.
- Do not use checkpointed MPS for scientific result claims.
- Do not run WT103, production training, large downloads, or main-machine experiments from this repo unless explicitly requested.
- Do not modify `neo.csv`; it lives outside this repo and clean MLX rows remain authoritative.
- Prefer small, reviewable task slices over broad changes.
- Preserve existing user changes. Do not revert unrelated files unless the user explicitly asks.
- Update docs when behavior, architecture, workflow, public commands, verification policy, or durable assumptions change.

## Planning Docs

- Read `docs/CONTEXT_BOOTSTRAP.md` when starting cold, after context compression, or when the task is broad.
- Treat `docs/ROADMAP.md` as broad priority direction, not a task queue.
- Treat `docs/PROGRESS.md` as the current checkpoint chain for the active large goal.
- Treat `docs/IMPLEMENTATION_PLAN.md` as the active execution contract and strict PR queue.
- Treat `docs/DEVLOG.md` as durable technical memory for implemented contracts, compatibility decisions, public behavior changes, verification gates, and workflow policy changes.
- Read `docs/DOCUMENTATION_STRUCTURE.md` before creating, restructuring, or simplifying docs.
- Keep completed technical outcomes out of the active queue after they are recorded in `docs/DEVLOG.md`.

## Test-First Policy

- For behavior changes, write or update the narrowest useful test before implementation.
- Treat the first test as the working contract for the task.
- Do not rewrite tests just to match accidental implementation behavior.
- If an up-front test must change after implementation starts, report the reason as one of:
  - requirement clarification
  - compatibility correction
  - invariant mismatch
  - test bug
  - overconstraint
- Docs-only, comment-only, rename-only, and mechanical formatting changes are exempt.

## Verification Rules

- `make check` is the standard local verification entrypoint.
- `just check` is an ergonomic wrapper around `make check`.
- GitHub Actions should call `make` targets so local and remote verification stay aligned.
- Optional hardware probes must stay skip-safe or opt-in:
  - MLX parity: `make mlx-parity`
  - MPS probe: `make mps-probe`
  - CUDA work: skip-safe until an Nvidia runner exists
- If verification cannot be run, explain why and name the command that should be run.

## Git And Change Hygiene

- Treat `main` as stable.
- Prefer short-lived branches for non-trivial implementation work.
- Use `justfile` helpers for branch, commit, push, and PR lifecycle when `just` is available.
- Keep GitHub lifecycle policy in `justfile`, not `Makefile`.
- Keep project correctness checks in `Makefile`, not `justfile`.
- Stage only files that belong to the current task.
- Direct commits to `main` are allowed only when the user explicitly asks for them or the task is a low-risk docs/workflow maintenance exception.

## Task Workflow

1. Read `docs/CONTEXT_BOOTSTRAP.md` for broad or cold-start tasks.
2. Identify the active goal in `docs/ROADMAP.md` and checkpoint state in `docs/PROGRESS.md`.
3. Identify the active slice in `docs/IMPLEMENTATION_PLAN.md`.
4. Select exactly one task packet from the strict queue.
5. Define public surface, inputs, outputs, failure behavior, persistence, compatibility, scope, and verification before editing behavior.
6. Add or update the first test before behavior-changing implementation.
7. Implement only the scoped files unless discovery proves the scope is wrong.
8. Run task-specific verification and `make check`.
9. Record durable decisions in `docs/DEVLOG.md`.
10. Update `docs/IMPLEMENTATION_PLAN.md` and `docs/PROGRESS.md` when their owned state changes.

## Response Requirements

Final responses should include:

- what changed
- key technical decisions or contract choices
- files changed
- commands run
- verification result and relevant output summary
- assumptions or interpretations that affected the work
- whether up-front tests changed after implementation started
- follow-up task if one should be added to the plan
