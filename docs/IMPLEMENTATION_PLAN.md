# Implementation Plan

Strict execution contract for Issue #2 backend parity work.

GitHub issue: https://github.com/ZFrankJ/neo-neuron/issues/2

## Current State

```text
main == origin/main
HEAD 62ebfc2 Add optional no-checkpoint MPS probe (#9)
```

MLX is the frozen scientific reference backend. Existing clean MLX result rows outside this repo remain authoritative.

The previous alignment queue established MLX reference parity, optimizer parity, public training-loop parity, checkpoint metadata guards, CI, and a seed optional MPS probe. The research goal continues with this ladder:

```text
MLX reference
  -> PyTorch CPU semantic bridge
  -> PyTorch MPS no-checkpoint parity
  -> PyTorch CUDA parity when Nvidia hardware exists
```

## Goal Exits

### MPS Parity Exit

MPS is parity-ready only when all of the following are true:

- MLX vs PyTorch CPU parity remains green on the established tiny reference suite.
- PyTorch CPU vs PyTorch MPS no-checkpoint forward, backward, optimizer, train/eval, and checkpoint-load probes pass within explicit tolerances.
- The MPS short trajectory keeps loss drift, parameter drift, gradient norms, and recurrent-state drift inside the documented parity envelope.
- The MPS memory probe shows flat or bounded memory behavior over the longest safe local probe.
- The trusted MPS path keeps `use_checkpoint: false`.
- Checkpointed MPS remains explicitly quarantined unless a later dedicated PR proves otherwise.
- No WT103, main-machine, production result, or `neo.csv` change is required for the parity claim.

### CUDA Transfer Exit

CUDA parity work is ready only after the MPS no-checkpoint ladder is stable and the same PyTorch CPU bridge can be reused unchanged for CUDA. CUDA must first pass full-precision, no-checkpoint, no-compile, no-fused-optimizer tests before speed features are considered.

## Global Rules

- Do not change MLX runtime semantics to match PyTorch.
- Do not weaken or remove existing MLX parity tests.
- Do not run WT103.
- Do not start production training scripts.
- Do not download large datasets.
- Do not modify `neo.csv`.
- Do not alter `configs/wt103/*` unless explicitly approved.
- Do not touch the main experiment machine.
- Do not write checkpoints or result artifacts outside pytest temp directories.
- Keep the passing PyTorch parity path on `use_checkpoint: false`.
- Keep checkpointed MPS outside scientific claims.
- Keep MPS/CUDA tests opt-in or skip-safe unless the runner is explicitly provisioned.
- Prefer test-first changes.
- Report exact local commands and GitHub Actions results in each PR body.

## Completed PRs

- PR #3: https://github.com/ZFrankJ/neo-neuron/pull/3
  - Merge commit: `7ab0634 Add MLX reference parity tests for Neo (#3)`
  - Established explicit RMSNorm epsilon, tiny forward/state/gradient/optimizer/checkpoint parity, and checkpoint closure binding coverage.
- PR #4: https://github.com/ZFrankJ/neo-neuron/pull/4
  - Merge commit: `ed72718 Add tiny CPU MLX training trajectory parity test (#4)`
  - Established a 100-step tiny PyTorch CPU vs MLX no-checkpoint trajectory.
- PR #5: https://github.com/ZFrankJ/neo-neuron/pull/5
  - Merge commit: `0ea3475 Add CI test workflow (#5)`
  - Added Ubuntu tests and macOS MLX parity CI.
- PR #6: https://github.com/ZFrankJ/neo-neuron/pull/6
  - Merge commit: `6d36d20 Add production-like optimizer parity (#6)`
  - Aligned the MLX-reference PyTorch AdamW path, weight decay buckets, and optimizer update order.
- PR #7: https://github.com/ZFrankJ/neo-neuron/pull/7
  - Merge commit: `16da73c Add public backend training loop parity (#7)`
  - Added public train/eval/checkpoint backend parity and aligned `grad_clip=0.0`.
- PR #8: https://github.com/ZFrankJ/neo-neuron/pull/8
  - Merge commit: `d7adeea Add checkpoint metadata guards (#8)`
  - Added aligned checkpoint metadata and guarded load/eval/resume paths.
- PR #9: https://github.com/ZFrankJ/neo-neuron/pull/9
  - Merge commit: `62ebfc2 Add optional no-checkpoint MPS probe (#9)`
  - Added opt-in no-checkpoint MPS diagnostics for tiny gradients and memory trend.

## Active PR Queue

Work through this queue in order. Do not start a later PR until the earlier PR is merged or explicitly skipped.

### PR #10: Backend Parity Audit Harness

- Branch:
  - `codex/fix/backend-parity-audit-harness`
- Goal:
  - Create one shared deterministic audit layer for backend parity measurements so MPS and CUDA work reuse the same metrics.
- Expected scope:
  - Test utilities or a small internal probe helper used only by tests/probes.
  - Structured report fields for backend pair, device pair, seed, model shape, `use_checkpoint`, loss diff, logits max diff, state diff, gradient diff, update diff, grad norm, NaN/Inf counts, and memory samples when available.
  - No production training behavior changes.
  - No MLX semantic changes.
- Required coverage:
  - CPU self-parity sanity test proves the auditor itself can pass exact or near-exact comparisons.
  - Existing MLX vs PyTorch CPU reference tests either use the helper directly or remain compatible with it.
  - MPS/CUDA unavailable paths skip cleanly.
- Exit criteria:
  - `make check` passes.
  - Existing GitHub Actions jobs pass.
  - The PR body shows one sample audit output from a tiny CPU or MLX/CPU run.
  - No WT103/main-machine/result/`neo.csv` changes.

### PR #11: MPS Single-Step Parity Envelope

- Branch:
  - `codex/fix/mps-single-step-parity-envelope`
- Goal:
  - Extend the optional MPS probe into a strict single-step CPU-vs-MPS parity envelope.
- Required coverage:
  - CPU vs MPS forward parity.
  - CPU vs MPS recurrent-state parity.
  - CPU vs MPS backward parity.
  - CPU vs MPS optimizer-update parity.
  - Explicit failure or xfail if `use_checkpoint=true` is attempted in the trusted MPS suite.
- Exit criteria:
  - Default CI still passes without MPS.
  - On Apple Silicon, `NEO_RUN_MPS_PROBE=1 python3 -m pytest -q tests/test_mps_no_checkpoint_probe.py` passes or reports a concrete blocker.
  - PR body reports max diffs and the largest offending parameter if any.

### PR #12: MPS Short Training Trajectory Parity

- Branch:
  - `codex/fix/mps-short-trajectory-parity`
- Goal:
  - Prove PyTorch MPS can follow PyTorch CPU over a short no-checkpoint training path after MLX vs PyTorch CPU parity is established.
- Required coverage:
  - Initial eval parity.
  - Stepwise loss drift.
  - Final eval parity.
  - Final parameter diff.
  - Final recurrent-state diff.
  - Gradient norm drift.
  - NaN/Inf guard.
- Exit criteria:
  - Default CI still passes without MPS.
  - Opt-in local MPS run reports pass/fail with exact tolerance values.
  - No WT103/main-machine/result/`neo.csv` changes.

### PR #13: MPS Endurance And Memory Slope Probe

- Branch:
  - `codex/fix/mps-memory-slope-probe`
- Goal:
  - Quantify whether no-checkpoint MPS memory behavior is flat, bounded, or growing under the longest safe local probe.
- Required coverage:
  - RSS samples.
  - MPS allocated-memory samples where PyTorch exposes them.
  - Loss samples.
  - Gradient norm samples.
  - NaN/Inf guard.
  - Memory slope classification: `flat`, `bounded_sawtooth`, `linear_growth`, or `superlinear_growth`.
- Exit criteria:
  - Probe can run locally without exhausting this machine.
  - Probe estimates memory growth per step and extrapolates cautiously without claiming WT103 validation.
  - CI remains skip-safe.

### PR #14: CUDA Parity Harness Preparation

- Branch:
  - `codex/fix/cuda-parity-harness-prep`
- Goal:
  - Prepare the parity harness so Nvidia CUDA can reuse the same CPU bridge later, without requiring CUDA hardware now.
- Required coverage:
  - CUDA skip-safe test markers or helpers.
  - `torch.cuda` device discovery/reporting.
  - CUDA parity test file that skips when CUDA is unavailable.
  - Full precision, no checkpoint, no `torch.compile`, no fused optimizer.
- Exit criteria:
  - `make check` passes on non-CUDA machines.
  - GitHub Actions Ubuntu and macOS jobs pass without CUDA.
  - PR body documents the exact command a CUDA machine should run later.

### PR #15: Optional Nvidia GPU CI Workflow

- Branch:
  - `codex/ci/optional-nvidia-gpu-ci`
- Goal:
  - Add a third optional GitHub Actions test lane for Nvidia CUDA parity, gated behind an explicitly provisioned GPU runner.
- Current runner reality:
  - Standard GitHub-hosted Ubuntu runners do not provide Nvidia CUDA hardware.
  - Nvidia CI is not necessary for the current MPS parity goal.
  - Do not make CUDA CI a required PR status until a GPU runner is intentionally provisioned.
- Exit criteria:
  - Existing Ubuntu and macOS CI stays green.
  - CUDA workflow is manual or otherwise impossible to run accidentally on CPU-only runners.
  - Branch protection does not require CUDA.

### PR #16: Branch Protection And Required Checks Policy

- Branch:
  - `codex/docs/branch-protection-policy`
- Goal:
  - Document intended branch protection so parity checks are hard to bypass.
- Required checks should initially be:
  - Ubuntu tests
  - macOS MLX parity
- Optional checks should initially be:
  - opt-in MPS probe
  - optional CUDA GPU workflow
- Exit criteria:
  - The policy matches the current runner reality and does not require unavailable GPU hardware.

## Parked Work

Do not hand these to an agent yet:

- WT103 reruns.
- Revalidating old PyTorch MPS result rows.
- Making MPS a production result backend.
- Re-enabling activation checkpointing for result production.
- Turning CUDA CI into a required branch protection check before an Nvidia runner exists.
- Editing `neo.csv` based on PyTorch runs.
- Introducing CUDA speed features such as AMP, TF32 changes, fused optimizers, `torch.compile`, or activation checkpointing before boring full-precision CUDA parity passes.

## Standard PR Body Template

```markdown
Follow-up to #2.

## Queue Item

- PR queue item: `<PR #10 / #11 / #12 / #13 / #14 / #15 / #16 name>`
- Depends on: `<merged PRs>`

## Summary

- `<what changed>`
- `<what parity risk this closes>`

## Scope

- `<files changed>`
- MLX runtime semantics unchanged.
- No WT103/main-machine/result/neo.csv changes.
- Trusted PyTorch parity path remains `use_checkpoint: false`.

## Verification

- `python3 -m pytest -q tests/test_mlx_reference_parity.py`: `<result>`
- `python3 -m pytest -q`: `<result>`
- `<optional MPS/CUDA command if this PR has one>`: `<result or skip reason>`
- `git diff --check`: `<result>`
- GitHub Actions Ubuntu tests: `<result or URL>`
- GitHub Actions macOS MLX parity: `<result or URL>`
- Optional CUDA workflow: `<not applicable / skipped / URL>`

## Remaining Follow-Up

- `<next queue item or none>`
```

## Merge Review Checklist

- Is the PR the next queue item, or has the queue been explicitly updated?
- Does the diff stay inside the stated scope?
- Are MLX runtime semantics unchanged?
- Is `use_checkpoint: false` preserved for passing PyTorch parity paths?
- Are checkpointed MPS claims still quarantined?
- Are tests small and deterministic?
- Are MPS/CUDA tests skip-safe unless explicitly provisioned?
- Are no WT103/main-machine/result/`neo.csv` changes present?
- Do local tests pass?
- Do GitHub Actions pass?
- Does the PR body report exact verification?

If any answer is no, do not merge until the issue is resolved or explicitly accepted.
