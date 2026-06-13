# DEVLOG

Durable technical memory for Neo. Keep active queues in `docs/IMPLEMENTATION_PLAN.md`; keep broad priorities in `docs/ROADMAP.md`.

## 2026-06-13 - MPS Short Training Trajectory Parity Probe

- Decision:
  - Extended the opt-in no-checkpoint MPS probe from single-step parity into a short CPU-vs-MPS training trajectory envelope.
- Why:
  - MPS parity needs evidence that forward, backward, recurrent state, optimizer, and eval behavior stay aligned over repeated no-checkpoint updates, not only one isolated optimizer step.
- Scope:
  - `tests/test_mps_no_checkpoint_probe.py`
  - `README.md`
  - `docs/IMPLEMENTATION_PLAN.md`
- Impact:
  - The opt-in MPS probe now reports initial eval parity, stepwise loss drift, final eval parity, final parameter diff, final recurrent-state diff, gradient norm drift, non-finite counts, and the largest offending parameter on failure.
  - The trajectory remains tiny, synthetic, no-checkpoint, and opt-in through `NEO_RUN_MPS_PROBE=1`; it is not WT103 or production-result validation.

## 2026-06-13 - MPS Single-Step Parity Envelope

- Decision:
  - Extended the opt-in no-checkpoint MPS probe into a strict CPU-vs-MPS single-step parity envelope.
- Why:
  - MPS parity claims need explicit forward, recurrent-state, backward, and optimizer-update measurements against the PyTorch CPU bridge before longer trajectory work.
- Scope:
  - `tests/test_mps_no_checkpoint_probe.py`
  - `README.md`
- Impact:
  - The opt-in MPS probe now reports structured audit fields for loss, logits, recurrent state, gradients, optimizer update, non-finite counts, and the largest offending update parameter on failure.
  - Trusted MPS parity probes reject `use_checkpoint=true`; checkpointed MPS remains outside scientific claims.
  - Default CI remains skip-safe without the MPS opt-in environment.

## 2026-06-13 - Backend Parity Audit Harness

- Decision:
  - Added a shared backend parity audit report helper for deterministic CPU, MLX, MPS, and future CUDA parity measurements.
- Why:
  - MPS and CUDA parity work need one structured measurement contract instead of bespoke per-test diff accounting.
- Scope:
  - `src/runtime/parity_audit.py`
  - `tests/test_backend_parity_audit.py`
  - `tests/test_mlx_reference_parity.py`
- Impact:
  - Audit reports now carry backend pair, device pair, seed, model shape, `use_checkpoint`, loss diff, logits diff, recurrent-state diff, gradient diff, update diff, gradient norm, non-finite counts, and memory samples.
  - CPU self-parity validates the auditor itself before optional hardware probes consume it.
  - Existing MLX-vs-PyTorch CPU reference tests now exercise the shared report helper without changing MLX runtime semantics.

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
