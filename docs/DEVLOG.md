# DEVLOG

Durable technical memory for Neo. Keep active queues in `docs/IMPLEMENTATION_PLAN.md`; keep broad priorities in `docs/ROADMAP.md`.

## 2026-07-11 - Baseline Alignment Queue Reopened

- Decision:
  - Reopened the active execution queue for baseline-alignment planning before new paper-facing result production.
  - Classified the current LSTM as a normalized/matched recurrent control, not a vanilla LSTM baseline.
  - Planned separate PRs for standard LSTM init controls, recurrent eval semantics, config/result labels, Transformer control strength, and LSTM optimizer grouping parity.
- Why:
  - Backend parity is locally complete, but paper claims also depend on baseline strength, evaluation semantics, and result provenance.
  - LSTM-specific best practices such as positive forget bias and orthogonal recurrent init have no direct Neo counterpart but can materially affect baseline strength.
  - Neo's move toward tanh activation requires explicit run labeling rather than relabeling older custom-activation results.
- Scope:
  - `docs/ROADMAP.md`
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`
- Impact:
  - Future agents should take one baseline-alignment PR at a time.
  - No WT103/main-machine experiment, `neo.csv` edit, or old-result reinterpretation is authorized by this planning update.

## 2026-07-09 - Main Training Mac Mini Constraints Record

- Decision:
  - Kept the portable macOS MLX parity lock unchanged and added a separate main training Mac mini constraints record.
  - Accepted `mlx==0.30.6` only as a machine-specific exception after `make mlx-parity` passes on that machine.
  - Kept PyTorch MPS rejected for scientific result runs because the local no-checkpoint memory slope probe failed.
- Why:
  - The PR #15 lock exists for reproducible portable parity and should not be loosened by a single-machine result.
  - The main training machine can still have a documented local stack if it proves parity locally.
- Scope:
  - `requirements-lock/constraints-mac-mini-training.txt`
  - `docs/hardware.md`
  - `Makefile`
  - `scripts/validate_torch_path.sh`
- Impact:
  - `make torch-validate` records Torch CPU, MLX, and MPS validation status without promoting MPS to a trusted production path.
  - MLX 0.30.6 remains non-portable and does not supersede `requirements-lock/constraints-macos-mlx.txt`.

## 2026-06-30 - macOS MLX Parity Dependency Constraints

- Decision:
  - Added macOS Apple Silicon parity constraints for the tested Torch, NumPy, MLX, and MLX-Metal dependency stack, and wired macOS MLX CI to install with those constraints.
- Why:
  - Fresh unpinned installs resolved to MLX 0.31.2 and failed the strict MLX reference parity suite with small but repeated numerical threshold misses.
  - Local matrix checks showed MLX 0.30.0 and 0.30.6 still failed, while MLX 0.29.4 passed with the current Torch 2.12.1 and NumPy 2.4.6 stack.
- Scope:
  - `requirements.txt`
  - `requirements-lock/constraints-macos-mlx.txt`
  - `.github/workflows/tests.yml`
  - `README.md`
  - `requirements.md`
- Impact:
  - macOS MLX parity installs are reproducible against the tested stack.
  - Linux CI remains on plain `requirements.txt` and is not constrained to macOS-only MLX packages.
  - MLX runtime semantics and parity thresholds are unchanged.

## 2026-06-13 - Active Parity PR Queue Closed

- Decision:
  - Closed the active parity PR queue after CUDA harness preparation and converted unavailable Nvidia CI work into standing policy instead of queued PR work.
- Why:
  - This repo has no Nvidia GPU access, and standard GitHub-hosted runners for individual repos are not treated as valid CUDA validation hardware.
  - Keeping optional Nvidia CI and branch-protection cleanup as active queue items would invite agents to open PRs that cannot prove the intended hardware contract.
- Scope:
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/PROGRESS.md`
  - `docs/ROADMAP.md`
- Impact:
  - The strict parity queue is empty until the user explicitly reopens it or a real Nvidia GPU runner exists.
  - Required CI should remain limited to available Ubuntu tests and macOS MLX parity.
  - CUDA parity claims still require `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py` to run, not skip, on a real Nvidia CUDA machine.

## 2026-06-13 - Local Parity Queue Completed Pending Nvidia Access

- Decision:
  - Marked the local MLX/PyTorch/MPS/CUDA-preparation parity queue as complete until real Nvidia GPU access exists.
- Why:
  - The repo now has skip-safe CUDA harness preparation, but this machine has no Nvidia GPU and standard GitHub-hosted runners for individual repos are not treated as Nvidia GPU runners.
- Scope:
  - `README.md`
  - `docs/ROADMAP.md`
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/CONTEXT_BOOTSTRAP.md`
  - `docs/hardware.md`
- Impact:
  - Reproducers must first run `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py` on a real Nvidia CUDA machine and confirm it does not skip before making CUDA parity or result claims.
  - No local implementation PR is currently active; optional Nvidia CI is not queued unless a real GPU runner exists or the user explicitly reopens that work.

## 2026-06-13 - CUDA Parity Harness Preparation

- Decision:
  - Added a skip-safe CUDA parity harness preparation path that reuses the PyTorch CPU bridge without requiring CUDA hardware in normal local or CI checks.
- Why:
  - CUDA parity should inherit the same boring baseline as MPS before any speed features are considered.
- Scope:
  - `src/runtime/parity_audit.py`
  - `tests/test_cuda_parity_harness.py`
  - `Makefile`
  - `justfile`
  - `README.md`
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/PROGRESS.md`
- Impact:
  - CUDA discovery now reports device availability, device count, current device metadata, PyTorch/CUDA runtime versions, and full-precision policy state without forcing CUDA hardware.
  - `make cuda-probe` runs the optional CUDA probe with `NEO_RUN_CUDA_PROBE=1`, while normal `make check` remains skip-safe on non-CUDA machines.
  - The trusted CUDA baseline is full precision, `use_checkpoint: false`, `use_compile: false`, no fused optimizer, and no TF32 speed path.

## 2026-06-13 - MPS Memory Slope Classification Probe

- Decision:
  - Extended the opt-in no-checkpoint MPS probe from a short trajectory check into an endurance-style memory slope classification report.
- Why:
  - MPS parity needs bounded memory evidence over a longer safe local probe before the result path can be trusted beyond tiny single-step behavior.
- Scope:
  - `tests/test_mps_no_checkpoint_probe.py`
  - `README.md`
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/PROGRESS.md`
- Impact:
  - The opt-in MPS probe now samples RSS, PyTorch MPS allocated memory when available, loss, and gradient norm across a 32-step no-checkpoint trajectory.
  - Memory behavior is classified as `flat`, `bounded_sawtooth`, `linear_growth`, or `superlinear_growth`, with passing claims limited to flat or bounded sawtooth behavior.
  - The probe remains synthetic, opt-in through `NEO_RUN_MPS_PROBE=1`, and does not validate WT103-scale training, checkpointed MPS, or production result rows.

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
