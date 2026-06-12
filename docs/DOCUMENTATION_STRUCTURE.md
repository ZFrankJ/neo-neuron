# Documentation Structure

This document defines the job of each planning and workflow file. If one fact appears in several docs, only one doc should own the detail; other docs should link or summarize it.

## Core Ownership

- `README.md`: user entrypoint, setup, backend usage, and current limitations.
- `AGENTS.md`: repo-local Codex and agent behavior contract.
- `justfile`: local Git, GitHub, branch, commit, push, and PR helpers.
- `Makefile`: project-owned verification entrypoint.
- `.github/workflows/tests.yml`: remote CI wrapper around `make` targets.
- `docs/ROADMAP.md`: short stable priority stack.
- `docs/PROGRESS.md`: checkpoint chain for the active large goal.
- `docs/IMPLEMENTATION_PLAN.md`: strict execution queue for the active roadmap item.
- `docs/DEVLOG.md`: durable completed technical, compatibility, workflow, and verification decisions.
- `docs/CONTEXT_BOOTSTRAP.md`: cold-start read order, discovery pass, and verification workflow.
- Topic docs such as `docs/training.md`, `docs/probing.md`, and `docs/hardware.md`: user-facing usage notes for that topic.

## Standard Sections

### `docs/ROADMAP.md`

Keep this short and stable:

1. purpose
2. priority order
3. one section per broad priority with status, why now, focus, exit condition, and expanded-plan pointers

Do not put task-level PR plans here.

### `docs/PROGRESS.md`

Keep this as checkpoint state for the active large goal:

1. active large goal
2. current checkpoint position
3. checkpoint chain
4. remaining scale estimate
5. update rules

Do not put strict task packets here.

### `docs/IMPLEMENTATION_PLAN.md`

Keep this as the active execution contract:

1. current state
2. goal exits
3. global rules
4. completed PRs
5. active PR queue
6. parked work
7. PR body template
8. merge review checklist

Completed PR detail should be compact. Durable outcomes belong in `docs/DEVLOG.md`.

### `docs/DEVLOG.md`

Keep reverse chronological entries with:

1. `Decision`
2. `Why`
3. `Scope`
4. `Impact`
5. optional `Supersedes`

Do not log routine bug fixes or repeat active queues.

### `docs/CONTEXT_BOOTSTRAP.md`

Keep this operational:

1. read order
2. discovery pass
3. change-location decision rules
4. feature wiring checklist
5. verification commands
6. doc update reminders

Do not duplicate the full roadmap or active queue.

## Command Workflow

The command workflow has three layers:

1. `justfile` owns Git, GitHub, branch, commit, push, and PR helpers.
2. `Makefile` owns project verification through `make check`.
3. `.github/workflows/tests.yml` calls `make` targets remotely.

Do not put project-specific lint, test, or build logic in `justfile` unless it is only an ergonomic wrapper around `make`.

Do not put GitHub branch or PR lifecycle policy in `Makefile`.
