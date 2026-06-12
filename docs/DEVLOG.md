# DEVLOG

Durable technical memory for Neo. Keep active queues in `docs/IMPLEMENTATION_PLAN.md`; keep broad priorities in `docs/ROADMAP.md`.

## 2026-06-12 - Codex Harness Workflow Adapted For Neo

- Decision:
  - Added a Neo-specific `codex-harness` workflow with `AGENTS.md`, `justfile`, `Makefile`, planning docs, and CI delegation to `make` targets.
- Why:
  - Future agents need a smooth repo-local workflow for MLX-reference, PyTorch CPU, MPS, and eventual CUDA parity work without rediscovering the intended process.
- Scope:
  - `AGENTS.md`
  - `Makefile`
  - `justfile`
  - `.github/workflows/tests.yml`
  - `README.md`
  - `docs/ROADMAP.md`
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/DOCUMENTATION_STRUCTURE.md`
  - `docs/CONTEXT_BOOTSTRAP.md`
- Impact:
  - `make check` is now the canonical local verification command.
  - GitHub Actions uses `make test` and `make mlx-parity` instead of duplicating raw pytest commands.
  - The strict active queue now lives inside `Neo/docs/IMPLEMENTATION_PLAN.md`.
  - MLX remains the frozen reference backend.
  - MPS parity remains no-checkpoint and opt-in; CUDA CI remains optional until Nvidia hardware is provisioned.

## 2026-06-12 - Backend Alignment Baseline Through PR #9

- Decision:
  - Treat PR #3 through PR #9 as the completed baseline for MLX reference parity and seed MPS diagnostics.
- Why:
  - The remaining goal is not generic documentation; it is the backend parity ladder toward MPS and later CUDA.
- Scope:
  - MLX reference parity
  - production-like optimizer parity
  - public backend training-loop parity
  - checkpoint metadata guards
  - optional no-checkpoint MPS probe
- Impact:
  - Existing clean MLX results remain authoritative.
  - PyTorch CPU is the semantic bridge.
  - PyTorch MPS must prove no-checkpoint parity before it can be trusted.
  - Checkpointed MPS remains quarantined.

## Entry Template

```markdown
## YYYY-MM-DD - <Technical Outcome>

- Decision:
- Why:
- Scope:
- Impact:
- Supersedes:
```
