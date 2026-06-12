# Context Bootstrap

Use this workflow when context is compressed, when starting from zero, or when the correct owner is not obvious.

## 1. Read Order

1. `AGENTS.md`
2. `README.md`
3. `docs/DOCUMENTATION_STRUCTURE.md`
4. `docs/ROADMAP.md`
5. `docs/PROGRESS.md`
6. `docs/IMPLEMENTATION_PLAN.md`
7. `docs/DEVLOG.md`
8. Topic docs relevant to the task:
   - `docs/training.md`
   - `docs/probing.md`
   - `docs/hardware.md`
   - `docs/quickstart.md`

## 2. Discovery Pass

Before editing, identify:

1. current branch and worktree state:
   - `just status` or `git status --short --branch`
2. active roadmap priority:
   - `docs/ROADMAP.md`
3. current checkpoint:
   - `docs/PROGRESS.md`
4. active queue item:
   - `docs/IMPLEMENTATION_PLAN.md`
5. current tests and CI:
   - `Makefile`
   - `.github/workflows/tests.yml`

## 3. Decide Change Location

Before adding code or docs, answer:

1. Which module owns the behavior?
2. Is the change MLX reference behavior, PyTorch parity behavior, or workflow-only?
3. Does it touch training, eval, checkpoint loading, optimizer state, backend adapters, or probes?
4. Does it affect scientific result interpretation?
5. What exact verification proves the change?

If ownership is unclear, stop and narrow the task before editing.

## 4. Feature Wiring Checklist

Use this before finalizing non-trivial changes:

- Define the public surface.
- Define input, output, error, persistence, and compatibility contracts.
- Add or update the narrowest useful test first when behavior changes.
- Keep CLI, config, checkpoint metadata, backend behavior, docs, and tests aligned when more than one surface is affected.
- Update `docs/IMPLEMENTATION_PLAN.md` when queue state changes.
- Update `docs/PROGRESS.md` only when checkpoint state changes.
- Add a `docs/DEVLOG.md` entry when a durable contract or workflow changes.
- Update `README.md` when setup, usage, or assumptions change.

## 5. Verify

Standard local verification:

```bash
make check
```

Specific checks:

```bash
make test
make mlx-parity
make mps-probe
```

`make mps-probe` is opt-in and requires Apple Silicon MPS. It is not evidence for checkpointed MPS or WT103-scale result production.

## 6. GitHub Workflow

For branch-based work:

```bash
just sync-main
just start-branch codex/fix/<topic>
make check
just stage <paths>
just commit "type(scope): summary"
just push
just pr-body-template /tmp/pr_body.md
just pr-create-from-file "type(scope): summary" /tmp/pr_body.md
```

For explicit direct-main exceptions:

```bash
make check
just stage <paths>
just commit-main "type(scope): summary"
just push-main
```

Use the direct-main path only when the user explicitly asks for it or the task is low-risk workflow/docs maintenance.
