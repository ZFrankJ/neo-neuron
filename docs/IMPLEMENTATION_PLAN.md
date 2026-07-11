# Implementation Plan

Strict execution contract for Neo backend parity and baseline-alignment work.

GitHub issue: https://github.com/ZFrankJ/neo-neuron/issues/2

## Current State

```text
branch codex/feat/gpt2-style-transformer-control from origin/main
base 1e7830d Merge pull request #20 from ZFrankJ/codex/docs/config-labels-activation-provenance
```

MLX is the frozen scientific reference backend. Existing clean MLX result rows outside this repo remain authoritative.

The dependency reproducibility maintenance slice constrains macOS Apple Silicon
MLX parity installs to a tested package stack. MLX runtime semantics and strict
parity thresholds remain unchanged.

The completed local backend alignment queue established MLX reference parity, optimizer parity, public training-loop parity, checkpoint metadata guards, CI, a seed optional MPS probe, a shared backend parity audit report helper, MPS short training trajectory parity, MPS memory slope classification, skip-safe CUDA harness preparation, and machine-specific Mac mini backend provenance. CUDA validation remains blocked on real Nvidia GPU access:

```text
MLX reference
  -> PyTorch CPU semantic bridge
  -> PyTorch MPS no-checkpoint parity
  -> PyTorch CUDA parity only after an Nvidia GPU preflight passes
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

CUDA parity work is prepared but not validated in this repo because no Nvidia GPU is currently available. Reproducers must first run `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py` on a real Nvidia CUDA machine and confirm the optional CUDA test runs instead of skips. CUDA must pass full-precision, no-checkpoint, no-compile, no-fused-optimizer tests before speed features are considered.

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
- PR #10: https://github.com/ZFrankJ/neo-neuron/pull/10
  - Merge commit: `d188b0e test(runtime): add backend parity audit harness (#10)`
  - Added a shared structured parity audit report helper for CPU, MLX, MPS, and future CUDA measurements.
- PR #11: https://github.com/ZFrankJ/neo-neuron/pull/11
  - Merge commit: `9781470 Merge pull request #11 from ZFrankJ/codex/fix/mps-single-step-parity-envelope`
  - Extended the optional MPS probe into a strict CPU-vs-MPS single-step parity envelope with checkpointed MPS rejection.
- PR #12: https://github.com/ZFrankJ/neo-neuron/pull/12
  - Merge commit: `c4f1f96 Merge pull request #12 from ZFrankJ/codex/fix/mps-short-trajectory-parity`
  - Extended the optional MPS probe into a short no-checkpoint CPU-vs-MPS training trajectory envelope.
- PR #13: https://github.com/ZFrankJ/neo-neuron/pull/13
  - Merge commit: `1d44dbf Merge pull request #13 from ZFrankJ/codex/fix/mps-memory-slope-probe`
  - Extended the optional MPS probe into a no-checkpoint endurance memory slope classification report.
- PR #14: https://github.com/ZFrankJ/neo-neuron/pull/14
  - Merge commit: `948bfe3 Merge pull request #14 from ZFrankJ/codex/fix/cuda-parity-harness-prep`
  - Added skip-safe CUDA discovery, full-precision CUDA baseline policy reporting, and an opt-in CUDA single-step parity harness.
- PR #15: https://github.com/ZFrankJ/neo-neuron/pull/15
  - Merge commit: `5792670 Merge pull request #15 from ZFrankJ/codex/fix/macos-mlx-dependency-reproducibility`
  - Added the portable macOS Apple Silicon MLX parity dependency lock and wired macOS MLX CI to use it.
- PR #17: https://github.com/ZFrankJ/neo-neuron/pull/17
  - Merge commit: `e84d6c9 Merge pull request #17 from ZFrankJ/codex/maint/torch-validation-preflight`
  - Added `make torch-validate` and a machine-specific main Mac mini constraints record without loosening the portable parity lock.
- PR #18: https://github.com/ZFrankJ/neo-neuron/pull/18
  - Merge commit: `d6df064 Merge pull request #18 from ZFrankJ/codex/feat/lstm-standard-init-controls`
  - Added opt-in standard-init PyTorch LSTM controls while preserving historical LSTM config and checkpoint behavior.
- PR #19: https://github.com/ZFrankJ/neo-neuron/pull/19
  - Merge commit: `58c484b Merge pull request #19 from ZFrankJ/codex/feat/recurrent-eval-semantics`
  - Added explicit compatibility-default block-reset and opt-in streaming recurrent evaluation regimes.
- PR #20: https://github.com/ZFrankJ/neo-neuron/pull/20
  - Merge commit: `1e7830d Merge pull request #20 from ZFrankJ/codex/docs/config-labels-activation-provenance`
  - Added exact WT2 parameter-count reporting labels and explicit activation provenance for future tanh runs.

## Active PR Queue

The strict parity queue is empty. The active queue is now baseline-alignment
planning for paper-facing results. These PRs must not run WT103, mutate
`neo.csv`, or reinterpret old results. They prepare configs, tests, docs, and
acceptance semantics for later approved experiment runs.

### PR #18: LSTM Standard-Init Strengthening

- Status:
  - merged as PR #18; retained here until the baseline-alignment queue closes

- Branch:
  - `codex/feat/lstm-standard-init-controls`
- Goal:
  - Add a stronger LSTM baseline path covering the three LSTM-specific concerns:
    positive forget-gate bias, orthogonal recurrent initialization, and dropout
    policy alignment.
- Rationale:
  - The current LSTM is a normalized recurrent control. That is acceptable, but
    zero forget bias, Xavier recurrent matrices, and extra inter-layer dropout
    can make it weaker than a well-tuned LSTM.
  - Neo has no direct forget gate or learned hidden-to-hidden recurrent matrix,
    so these are LSTM-specific best-practice controls, not one-to-one Neo
    settings.
- Scope:
  - Add explicit LSTM init controls such as `forget_bias_init` and
    `recurrent_init`.
  - Add or document dropout-policy controls so LSTM-only inter-layer dropout is
    intentional rather than accidental.
  - Add tests that inspect initialized LSTM gate biases and recurrent matrix
    properties on tiny models.
  - Prefer new configs or explicit labels over silently changing old result
    configs.
- Exit criteria:
  - Existing configs remain loadable.
  - New standard-init LSTM path is covered by tests.
  - Docs distinguish `RMSNorm-LSTM matched control` from `standard-init
    RMSNorm-LSTM`.
  - `make check` passes.

### PR #19: Recurrent Eval Semantics

- Status:
  - merged as PR #19; retained here until the baseline-alignment queue closes

- Branch:
  - `codex/feat/recurrent-eval-semantics`
- Goal:
  - Decide and implement how recurrent models report block-reset evaluation
    versus streaming-state evaluation.
- Rationale:
  - Training streams state across contiguous batches, but current eval resets
    state per non-overlapping evaluation batch. That is valid if documented, but
    it may understate recurrent models' sequence-memory performance.
- Scope:
  - Add an explicit config/CLI metric option for eval mode, for example
    `eval_regime: block_reset | streaming`.
  - Keep current behavior as `block_reset` unless a deliberate compatibility
    decision changes it.
  - Add focused tests on a tiny recurrent model or sentinel model to prove state
    reset/streaming behavior.
  - Update result docs so future tables state which eval regime is used.
- Exit criteria:
  - Both eval regimes are deterministic and tested.
  - Existing checkpoints/configs keep their current eval interpretation.
  - `make check` passes.

### PR #20: Config Labels And Activation Provenance

- Status:
  - merged as PR #20; retained here until the baseline-alignment queue closes

- Branch:
  - `codex/docs/config-labels-activation-provenance`
- Goal:
  - Clean up config/result labels before new runs.
- Rationale:
  - WT2 config names such as `*_6m` and `lstm_25m` do not match current
    parameter counts. Neo is also moving toward tanh activation, so labels must
    distinguish tanh runs from older `id4`/`id5` custom-activation runs.
- Scope:
  - Retain WT2 config paths for compatibility, but document them as legacy
    labels with exact parameter counts and small/large reporting names.
  - Require exact parameter counts in paper-facing result tables and PR bodies.
  - Use `activation_id: tanh` and activation-bearing run tags for future WT103
    Neo templates; do not relabel old `id4`/`id5` runs as tanh.
  - Update scripts/notebook references only if paths are renamed.
- Exit criteria:
  - Config labels no longer imply inaccurate parameter counts.
  - Activation provenance is explicit for old and new Neo runs.
  - `make check` passes.

### PR #21: GPT-Style Transformer Control

- Status:
  - implemented on `codex/feat/gpt2-style-transformer-control`; pending review and merge

- Branch:
  - `codex/feat/gpt2-style-transformer-control`
- Goal:
  - Strengthen the Transformer comparison path or explicitly demote it to a
    lightweight internal control.
- Rationale:
  - The current Transformer is hand-rolled with learned absolute positions and
    unfused attention. It is useful for smoke comparisons, but weak for claims
    against modern Transformer baselines.
- Scope:
  - Add a GPT-2-style internal baseline when feasible: pre-norm blocks, causal
    attention using PyTorch/MLX-supported optimized primitives where available,
    GPT-style initialization/residual scaling, and clear config labels.
  - If implementation scope is too large, document the current Transformer as an
    internal control only and park stronger Transformer comparisons.
  - Keep MLX/Torch behavior aligned or explicitly mark backend support limits.
- Exit criteria:
  - Transformer baseline status is unambiguous in docs.
  - New code, if added, has focused shape/causality/checkpoint tests.
  - `make check` passes.

### PR #22: LSTM Optimizer Grouping Parity Guard

- Branch:
  - `codex/fix/lstm-optimizer-grouping-guard`
- Goal:
  - Fix or prove the Torch-vs-MLX optimizer grouping edge case for LSTM.
- Rationale:
  - Normal Torch optimizer grouping detects LSTM recurrent parameters with
    names under `lstm.*`. The MLX-reference Torch optimizer path checks
    `lstm_layers.*`, which matches MLX naming but not Torch LSTM naming. If a
    Torch LSTM is trained with `reference_backend: mlx`, its recurrent weights
    may receive projection weight decay instead of `recurrent_weight_decay`.
- Scope:
  - Add a focused unit test that builds a Torch LSTM under MLX-reference
    optimizer settings and verifies recurrent/projection/embedding/norm decay
    buckets.
  - Fix grouping to recognize both Torch `lstm.*` and MLX-style
    `lstm_layers.*` names where appropriate.
  - Keep existing MLX parity behavior unchanged.
- Exit criteria:
  - The test fails before the fix and passes after it.
  - No optimizer behavior changes for Neo or Transformer unless explicitly
    covered by tests.
  - `make check` passes.

The next CUDA step remains external validation, not normal local development.
Before opening CUDA-result or CUDA-CI work, confirm an
Nvidia CUDA environment with:

```bash
NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py
```

If this command skips because CUDA is unavailable, CUDA parity has not been validated. Standard GitHub-hosted runners for individual repos are not treated as Nvidia GPU runners for this project.

## Standing CI And Branch Protection Policy

Required checks should remain limited to currently available runners:

- Ubuntu tests
- macOS MLX parity

Do not add required CUDA CI while this project has no Nvidia GPU runner. Standard GitHub-hosted Ubuntu runners for individual repos do not provide Nvidia CUDA hardware. A future CUDA workflow should be manual-only or optional until a real Nvidia runner is provisioned and explicitly approved.

Optional hardware probes remain local/manual:

- MPS: `NEO_RUN_MPS_PROBE=1 python3 -m pytest -q tests/test_mps_no_checkpoint_probe.py`
- CUDA: `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py`

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

- PR queue item: `<PR #18 / #19 / #20 / #21 / #22 name>`
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
