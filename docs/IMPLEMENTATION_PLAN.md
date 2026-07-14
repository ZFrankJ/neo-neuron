# Implementation Plan

Strict execution contract for Neo backend parity and baseline-alignment work.

Previous tracking issue: https://github.com/ZFrankJ/neo-neuron/issues/2

Issue #2 is closed. PR #28 completed the local LSTM four-way correction queue;
open a new tracking issue only when explicitly requested.

## Current State

```text
completed packet PR 1: https://github.com/ZFrankJ/neo-neuron/pull/29
next active packet PR 2: WT103 50M-recurrent-core LSTM diagnostic profiles
```

MLX is the frozen scientific reference backend. Existing clean MLX result rows outside this repo remain authoritative.

The dependency reproducibility maintenance slice constrains macOS Apple Silicon
MLX parity installs to a tested package stack. MLX runtime semantics and strict
parity thresholds remain unchanged.

The completed local Neo backend alignment queue established same-weight MLX
reference parity, optimizer parity, public training-loop parity, checkpoint
metadata guards, CI, a seed optional MPS probe, a shared backend parity audit
report helper, MPS short training trajectory parity, MPS memory slope
classification, skip-safe CUDA harness preparation, and machine-specific Mac
mini backend provenance. The LSTM extension now includes mapped forward,
gradient, optimizer, trajectory, checkpoint, public-loop scheduler, and
explicit trial-profile contracts. CUDA validation remains blocked on real
Nvidia GPU access:

```text
MLX reference
  -> PyTorch CPU semantic bridge
  -> PyTorch MPS no-checkpoint parity
  -> PyTorch CUDA parity only after an Nvidia GPU preflight passes
```

A four-path audit after the LSTM queue closed separated three contracts that
must not be conflated:

1. mapped-weight MLX/Torch semantic parity
2. fresh backend-local training profiles
3. Neo/LSTM result fairness under parameter-matched or geometry-matched designs

The audit keeps clean MLX results authoritative under their historical labels,
but rejects plain `--backend torch` overrides of the historical WT103 configs as
aligned experiments. It also identifies LSTM inter-layer dropout as an
unresolved depth-dependent regularization confound.

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

### LSTM Four-Way Alignment Exit

LSTM is alignment-ready only when all of the following are true:

- Same mapped weights produce matching MLX/Torch LSTM logits and recurrent
  state for `none`, `layernorm`, and `rmsnorm` recurrent normalization.
- The aligned Torch path has one trainable effective gate bias, matching MLX,
  without silently changing legacy split-bias checkpoint interpretation.
- RMSNorm epsilon is explicit and equal to `1e-5` in the aligned profile.
- Loss, gradients, one optimizer update, a short fixed-batch trajectory, and
  same-backend resume pass explicit parity tolerances without relaxing the Neo
  contract.
- MLX and Torch honor the same explicit LSTM layer-dropout and standard-init
  controls for the new profile while missing controls preserve historical
  backend behavior.
- The MLX-reference Torch public loop matches MLX warmup/cosine update timing.
- A checked-in profile freezes the single-bias, `rmsnorm_eps: 1e-5`,
  no-layer-dropout, standard-init, no-checkpoint, streaming-evaluation contract
  with equal backend trainable parameter counts.
- Historical MLX and Torch LSTM runs retain backend-specific provenance and are
  not relabeled as aligned or standard-init runs.
- No WT103 run or `neo.csv` edit is needed to satisfy this exit.

### WT103 Revalidation Exit

WT103 revalidation is ready to expand beyond one diagnostic only when all of the
following are true:

- Historical `configs/wt103/neo_*.yaml` and `configs/wt103/lstm_*.yaml` paths,
  run tags, checkpoints, and result labels remain unchanged.
- Neo MLX rejects unsupported explicit `rmsnorm_eps` values rather than silently
  recording a value it does not execute.
- The committed Neo parity matrix covers the current tanh baseline, and parity
  tests do not leak MLX device state into later test modules.
- New LSTM diagnostic configs make `reference_backend: mlx`, single bias,
  `rmsnorm_eps: 1e-5`, no checkpointing, streaming evaluation, and layer-dropout
  policy explicit.
- The matched profile preserves native MLX initialization by omitting standard
  initialization keys; the standard-init fallback explicitly selects positive
  forget bias and orthogonal recurrent initialization.
- The first run is only the `d_model=790`, `n_layers=10` MLX LSTM: approximately
  50M recurrent-core and 60M total parameters, configured for 12 epochs and
  reviewed after epoch 4.
- Broader LSTM scaling is authorized only if the diagnostic shows meaningful
  recovery; clean MLX Neo checkpoints are reevaluated rather than retrained.

## Global Rules

- Do not change MLX runtime semantics to match PyTorch.
- New MLX LSTM capabilities may be added only as explicit opt-in controls whose
  absence preserves historical MLX behavior; they are new experimental
  profiles, not silent parity repairs.
- Do not weaken or remove existing MLX parity tests.
- Do not run WT103 from an implementation PR or before the active diagnostic
  packet is merged and explicitly started by the user.
- Do not start production training scripts.
- Do not download large datasets.
- Do not modify `neo.csv`.
- Do not alter historical `configs/wt103/*` files. The current approval permits
  only new profile paths named in the active queue.
- Do not reuse a historical run tag for a new LSTM bias, normalization,
  dropout, initialization, or scheduler contract.
- Do not touch the main experiment machine from implementation PRs. The
  diagnostic starts only after the config packet merges and the user explicitly
  starts or authorizes it.
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
- PR #21: https://github.com/ZFrankJ/neo-neuron/pull/21
  - Merge commit: `db821b4 Merge pull request #21 from ZFrankJ/codex/feat/gpt2-style-transformer-control`
  - Added an opt-in GPT-2-style Transformer control with aligned Torch/MLX causal checkpoint behavior.
- PR #22: https://github.com/ZFrankJ/neo-neuron/pull/22
  - Merge commit: `9b00292 Merge pull request #22 from ZFrankJ/codex/fix/lstm-optimizer-grouping-guard`
  - Aligned MLX-reference Torch LSTM optimizer grouping across backend-specific recurrent parameter names.
- PR #23: https://github.com/ZFrankJ/neo-neuron/pull/23
  - Merge commit: `2ca64ea Merge pull request #23 from ZFrankJ/codex/fix/lstm-effective-bias-contract`
  - Added an explicit single-effective-bias Torch LSTM mode matching MLX trainable parameter and update semantics.
- PR #24: https://github.com/ZFrankJ/neo-neuron/pull/24
  - Merge commit: `4b26e91 Merge pull request #24 from ZFrankJ/codex/fix/lstm-forward-checkpoint-parity`
  - Added deterministic MLX/Torch LSTM forward, recurrent-state, loss, and checkpoint parity with explicit RMSNorm epsilon handling.
- PR #25: https://github.com/ZFrankJ/neo-neuron/pull/25
  - Merge commit: `807a080 Merge pull request #25 from ZFrankJ/codex/test/lstm-training-trajectory-parity`
  - Added deterministic LSTM gradient, optimizer, short fixed-batch trajectory, and backend-native optimizer-resume parity.
- PR #26: https://github.com/ZFrankJ/neo-neuron/pull/26
  - Merge commit: `df19d7f Merge pull request #26 from ZFrankJ/codex/feat/mlx-lstm-aligned-controls`
  - Added explicit matched-dropout and standard-init LSTM controls on MLX while preserving missing-key native initialization.
- PR #27: https://github.com/ZFrankJ/neo-neuron/pull/27
  - Merge commit: `61b56c5 Merge pull request #27 from ZFrankJ/codex/fix/mlx-reference-scheduler-timing`
  - Aligned MLX-reference Torch warmup/cosine timing and added deterministic public-loop LSTM parity.
- PR #28: https://github.com/ZFrankJ/neo-neuron/pull/28
  - Merge commit: `891550d Merge pull request #28 from ZFrankJ/codex/docs/lstm-aligned-trial-profile`
  - Froze the checked-in standard-init no-layer-dropout LSTM trial profile, exact backend parameter counts, schedule, evaluation regime, and provenance labels.
- PR #29: https://github.com/ZFrankJ/neo-neuron/pull/29
  - Hardened the frozen Neo MLX RMSNorm epsilon contract, added tanh to the
    mapped-weight parity matrix, and isolated MLX default-device state in tests.

## Active PR Queue

Execute exactly one packet at a time in this order.

### PR 2 - WT103 50M-Recurrent-Core LSTM Diagnostic Profiles

- Depends on:
  - PR #29 merged
- Purpose:
  - create the minimum explicit config surface needed to test whether repeated
    LSTM inter-layer dropout caused the observed depth degradation
- New paths only:
  - `configs/wt103/lstm_60m_matched_no_layer_dropout.yaml`
  - `configs/wt103/lstm_60m_standard_init_no_layer_dropout.yaml`
- Shared contract:
  - MLX training backend and `reference_backend: mlx`
  - `d_model: 790` and `n_layers: 10`, matching the recorded approximately 50M
    recurrent-core / 60M-total historical LSTM geometry
  - exactly `60,024,343` trainable parameters on MLX and aligned single-bias
    Torch construction
  - `lstm_bias_mode: single`
  - `recurrent_norm: rmsnorm`, `recurrent_norm_place: all`, and
    `rmsnorm_eps: 1e-5`
  - `lstm_layer_dropout: 0.0` while retaining output `dropout: 0.1`
  - `use_checkpoint: false`
  - `eval_regime: streaming`
  - 12 epochs, existing WT103 optimizer/scheduler values, fresh start, per-epoch
    checkpoints, seed `42`, and unique provenance-bearing run tags
- Profile difference:
  - matched profile omits `forget_bias_init` and `recurrent_init`, preserving
    native MLX initialization
  - standard-init profile sets `forget_bias_init: 1.0` and
    `recurrent_init: orthogonal`
  - Torch construction of the matched profile is a compatibility and evaluation
    check, not a fresh-run initialization-equivalence claim
- Required tests:
  - both configs construct on MLX and Torch
  - both configs expose the intended trainable counts and policy fields
  - historical WT103 config contracts remain unchanged
  - run tags cannot collide with historical artifacts
  - the config documentation freezes the epoch-4 validation baselines and gate
    before either run starts
- Excluded:
  - Neo retraining configs
  - 20M/30M/50M/70M/80M/90M total-parameter LSTM profile matrices
  - starting either training run
- Verification:
  - config-contract tests
  - `make lstm-parity`
  - `make check`
- Exit:
  - both diagnostic profiles are reviewable and runnable, with historical paths
    untouched

### Experiment Gate - Not A Code PR

- Start only the matched-no-layer-dropout `d_model=790`, `n_layers=10` MLX LSTM.
- Keep `epochs: 12`; do not shorten the schedule when planning to review at
  epoch 4.
- Use streaming validation PPL for the decision. Do not inspect test PPL until a
  profile and checkpoint have been selected.
- The frozen historical epoch-4 streaming validation baselines from
  `experiments/wt103/eval_regimes_epoch4_20260714.csv` are `84.54` for the same
  `d_model=790`, `n_layers=10` geometry and `82.57` for the preceding
  `d_model=790`, `n_layers=8` point.
- Continue the matched profile to epoch 12 when epoch-4 streaming validation PPL
  is at most `83.54`, a predeclared improvement of at least one PPL point over
  the same-geometry historical run. A value at most `82.57` is classified as
  full restoration of monotonic scaling at this checkpoint.
- If matched epoch-4 validation PPL is above `83.54`, stop it and run only the
  standard-init fallback to epoch 4 under the same gate.
- If the standard-init fallback also remains above `83.54`, stop and classify
  the 50M-recurrent-core recovery attempt as negative; do not rebuild the scale.
- `83.54` is an operational resource-allocation gate, not a statistical
  significance threshold. A successful one-seed diagnostic selects the next
  profile; it does not establish a paper-facing superiority claim.
- Any NaN or infinity is classified as an optimization failure and triggers a
  stop. A checkpoint or data failure invalidates the run and must be repaired
  before interpretation.
- Do not run both profiles concurrently, rebuild the scale, or retrain Neo before
  this gate is classified.
- A later result-record update is a separate, evidence-driven packet.

The next CUDA step remains external validation, not normal local development.
Before opening CUDA-result or CUDA-CI work, confirm an Nvidia CUDA environment
with:

```bash
NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py
```

If this command skips because CUDA is unavailable, CUDA parity has not been
validated. Standard GitHub-hosted runners for individual repos are not treated
as Nvidia GPU runners for this project.

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

- Full WT103 scaling reruns before the 50M-recurrent-core diagnostic gate is classified.
- Revalidating old PyTorch MPS result rows.
- Making MPS a production result backend.
- Re-enabling activation checkpointing for result production.
- Turning CUDA CI into a required branch protection check before an Nvidia runner exists.
- Editing `neo.csv` based on PyTorch runs.
- Exact cross-backend replay of NumPy/Torch random batches or stochastic
  dropout masks; deterministic parity uses fixed batches and dropout disabled,
  while paper variance uses repeated backend-local seeds.
- Aligning the divergent `tbptt_len < block_size` optimizer-step contract until
  a future approved config actually needs shorter TBPTT chunks.
- Adding WT103 profile matrices beyond the two approved 50M-recurrent-core diagnostic
  paths before the diagnostic gate is classified.
- Making a paper-facing recovered-scaling claim before a separate repeated-seed
  variance packet is approved and completed.
- Introducing CUDA speed features such as AMP, TF32 changes, fused optimizers, `torch.compile`, or activation checkpointing before boring full-precision CUDA parity passes.

## Standard PR Body Template

```markdown
Neo explicitly approved work packet.

## Queue Item

- PR queue item: `<future explicitly approved packet>`
- Depends on: `<merged PRs>`

## Summary

- `<what changed>`
- `<what parity risk this closes>`

## Scope

- `<files changed>`
- Historical MLX behavior is unchanged when new controls are absent.
- Historical WT103 configs and run tags are unchanged; any new WT103 path is an
  explicitly queued profile.
- No main-machine process, result, or `neo.csv` changes.
- Trusted PyTorch parity path remains `use_checkpoint: false`.

## Verification

- `make lstm-parity PYTHON=.venv/bin/python`: `<result or not yet available>`
- `make mlx-parity PYTHON=.venv/bin/python`: `<result>`
- `make check PYTHON=.venv/bin/python`: `<result>`
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
- Does missing-key behavior preserve historical MLX and Torch profiles?
- Are new LSTM controls explicit and provenance-visible rather than inferred
  silently from the backend?
- Does the PR preserve or deliberately migrate the single effective-bias and
  `rmsnorm_eps: 1e-5` aligned contracts?
- Is `use_checkpoint: false` preserved for passing PyTorch parity paths?
- Are checkpointed MPS claims still quarantined?
- Are tests small and deterministic?
- Are MPS/CUDA tests skip-safe unless explicitly provisioned?
- Are no WT103 runs, main-machine process changes, result edits, or `neo.csv`
  changes present?
- If WT103 templates are touched, are they new active-queue paths with unique
  run tags while every historical path remains unchanged?
- Do local tests pass?
- Do GitHub Actions pass?
- Does the PR body report exact verification?

If any answer is no, do not merge until the issue is resolved or explicitly accepted.
