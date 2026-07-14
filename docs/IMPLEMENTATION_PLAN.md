# Implementation Plan

Strict execution contract for Neo backend parity and baseline-alignment work.

Previous tracking issue: https://github.com/ZFrankJ/neo-neuron/issues/2

Issue #2 is closed. The LSTM four-way correction queue starts from live `main`;
open a new tracking issue only when explicitly requested.

## Current State

```text
branch codex/fix/lstm-forward-checkpoint-parity from origin/main
base 2ca64ea Merge pull request #23 from ZFrankJ/codex/fix/lstm-effective-bias-contract
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
mini backend provenance. A later four-way audit found that this contract does
not yet extend to LSTM. CUDA validation remains blocked on real Nvidia GPU
access:

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
- Historical MLX and Torch LSTM runs retain backend-specific provenance and are
  not relabeled as aligned or standard-init runs.
- No WT103 run or `neo.csv` edit is needed to satisfy this exit.

## Global Rules

- Do not change MLX runtime semantics to match PyTorch.
- New MLX LSTM capabilities may be added only as explicit opt-in controls whose
  absence preserves historical MLX behavior; they are new experimental
  profiles, not silent parity repairs.
- Do not weaken or remove existing MLX parity tests.
- Do not run WT103.
- Do not start production training scripts.
- Do not download large datasets.
- Do not modify `neo.csv`.
- Do not alter `configs/wt103/*` unless explicitly approved.
- Do not reuse a historical run tag for a new LSTM bias, normalization,
  dropout, initialization, or scheduler contract.
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
- PR #21: https://github.com/ZFrankJ/neo-neuron/pull/21
  - Merge commit: `db821b4 Merge pull request #21 from ZFrankJ/codex/feat/gpt2-style-transformer-control`
  - Added an opt-in GPT-2-style Transformer control with aligned Torch/MLX causal checkpoint behavior.
- PR #22: https://github.com/ZFrankJ/neo-neuron/pull/22
  - Merge commit: `9b00292 Merge pull request #22 from ZFrankJ/codex/fix/lstm-optimizer-grouping-guard`
  - Aligned MLX-reference Torch LSTM optimizer grouping across backend-specific recurrent parameter names.
- PR #23: https://github.com/ZFrankJ/neo-neuron/pull/23
  - Merge commit: `2ca64ea Merge pull request #23 from ZFrankJ/codex/fix/lstm-effective-bias-contract`
  - Added an explicit single-effective-bias Torch LSTM mode matching MLX trainable parameter and update semantics.

## Active PR Queue

Execute exactly one packet at a time. Expected PR numbers are based on the live
remote state after PR #23 merged. PR #24 is implemented on the branch below;
recheck GitHub before creating each PR because numbers can drift.

### PR #24: LSTM Forward And Checkpoint Parity

- Status:
  - implemented on `codex/fix/lstm-forward-checkpoint-parity`; pending review and merge
- Branch:
  - `codex/fix/lstm-forward-checkpoint-parity`
- Goal:
  - Establish the missing deterministic MLX/Torch LSTM forward contract.
- Public contract:
  - Add explicit LSTM `rmsnorm_eps` handling.
  - The aligned profile uses `rmsnorm_eps: 1e-5`.
  - Legacy Torch configs that omit the field retain their historical epsilon
    interpretation; MLX may accept missing or `1e-5`, but must reject an
    unsupported explicit value instead of silently ignoring it.
- Scope:
  - Add same-weight forward, recurrent-state, and loss parity for one and
    multiple layers with `none`, `layernorm`, and `rmsnorm`.
  - Cover MLX-to-Torch and Torch-to-MLX model checkpoint conversion.
  - Add LSTM checkpoint metadata guards for bias mode, recurrent norm,
    norm placement, and RMSNorm epsilon. Missing fields remain legacy warnings.
  - Add a dedicated `make lstm-parity` target and keep it skip-safe without
    MLX.
- Exit criteria:
  - RMSNorm mapped-weight logits return to the established small numerical
    envelope without tolerance relaxation.
  - Both checkpoint directions preserve evaluation loss.
  - Existing Neo parity remains green.
  - `make check` and `make lstm-parity` pass.

### PR #25: LSTM Gradient And Trajectory Parity

- Status:
  - queued after PR #24
- Branch:
  - `codex/test/lstm-training-trajectory-parity`
- Goal:
  - Extend LSTM parity from inference into training semantics.
- Contract:
  - Use the explicit single-bias, `rmsnorm_eps: 1e-5`,
    `reference_backend: mlx`, no-dropout, fixed-batch, no-checkpoint profile.
  - Exact random-batch or dropout-mask replay is outside this deterministic
    contract.
- Scope:
  - Compare loss, every mapped gradient, gradient norm, one Adam update,
    recurrent state, and a short fixed-batch trajectory.
  - Verify recurrent/projection/embedding/norm decay roles through the public
    optimizer path.
  - Cover same-backend optimizer resume. Cross-backend optimizer-state mapping
    remains unsupported and must be explicit.
  - Establish tolerances from measured evidence; do not weaken Neo thresholds.
- Exit criteria:
  - Gradient and update parity catches the old doubled-bias behavior.
  - The short trajectory remains within documented thresholds.
  - No non-finite values occur.
  - `make check` and `make lstm-parity` pass.

### PR #26: Cross-Backend LSTM Baseline Controls

- Status:
  - queued after PR #25
- Branch:
  - `codex/feat/mlx-lstm-aligned-controls`
- Goal:
  - Make the matched and strong LSTM profiles available on the MLX result
    backend without changing historical MLX defaults.
- Public contract:
  - MLX and Torch both honor explicit `lstm_layer_dropout`,
    `forget_bias_init`, and `recurrent_init`.
  - Missing keys preserve native historical behavior on each backend.
  - The aligned matched profile uses `lstm_layer_dropout: 0.0`.
  - The strong profile additionally uses `forget_bias_init: 1.0` and
    `recurrent_init: orthogonal`.
- Scope:
  - Remove the current MLX rejection only for implemented, tested controls.
  - Define the explicit standard-init profile completely: Xavier input
    matrices, selected recurrent initialization, zero non-forget biases, and
    the configured forget-bias value.
  - Test initialization invariants independently on both backends. Equal random
    tensors from equal seeds are not required.
  - Preserve missing-key MLX native uniform/random-bias initialization exactly.
  - Do not add or alter WT103 configs in this PR.
- Exit criteria:
  - Neo and matched LSTM have no inter-layer dropout under the explicit matched
    profile.
  - Standard-init invariants hold on both backends.
  - Historical configs construct with unchanged defaults.
  - `make check` and `make lstm-parity` pass.

### PR #27: MLX-Reference Warmup And Public-Loop Parity

- Status:
  - queued after PR #26
- Branch:
  - `codex/fix/mlx-reference-scheduler-timing`
- Goal:
  - Align the Torch MLX-reference public loop with MLX's first-update
    warmup/cosine timing.
- Compatibility:
  - Apply the aligned timing only to the explicit MLX-reference Torch path.
    Native Torch behavior and historical configs remain unchanged.
- Scope:
  - Add a fail-first scheduler test proving that MLX update one uses warmup step
    one while current Torch update one uses zero learning rate.
  - Add a public-loop LSTM parity case with cosine warmup, fixed streaming
    batches, deterministic initialization, and explicit aligned controls.
  - Keep random batch selection and stochastic dropout outside exact
    cross-backend trajectory claims.
  - Keep `tbptt_len < block_size` out of scope because checked-in configs do
    not activate that divergent branch.
- Exit criteria:
  - Per-update learning rates and public-loop metrics match the MLX contract.
  - Native Torch scheduling remains covered.
  - `make check` and `make lstm-parity` pass.

### PR #28: Aligned Profile And Trial Readiness

- Status:
  - queued after PR #27; WT103 template work remains approval-gated
- Branch:
  - `codex/docs/lstm-aligned-trial-profile`
- Goal:
  - Freeze the exact profile and provenance needed for a small alignment trial,
    then close the correction queue without starting production runs.
- Scope:
  - Add a lightweight checked-in alignment config or fixture containing
    `reference_backend: mlx`, `lstm_bias_mode: single`,
    `rmsnorm_eps: 1e-5`, `lstm_layer_dropout: 0.0`, and the selected init
    controls.
  - Document three distinct labels: legacy MLX LSTM, matched no-layer-dropout
    LSTM, and standard-init no-layer-dropout LSTM.
  - Record exact backend parameter counts and required evaluation regime.
  - Update README, training docs, DEVLOG, progress, and this queue to the final
    implemented behavior.
  - Do not modify existing WT103 configs or run tags. Propose new WT103 paths
    and tags, but add them only after explicit user approval.
  - Do not run WT103, touch the main training machine, or edit `neo.csv`.
- Exit criteria:
  - `make check`, `make mlx-parity`, and `make lstm-parity` pass.
  - Historical artifacts remain unambiguously labeled.
  - The next action is a separate approved result-production and variance plan,
    not another hidden implementation change.

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

- WT103 reruns.
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
- Adding aligned WT103 templates or run tags before the explicit approval gate
  in PR #28.
- Introducing CUDA speed features such as AMP, TF32 changes, fused optimizers, `torch.compile`, or activation checkpointing before boring full-precision CUDA parity passes.

## Standard PR Body Template

```markdown
LSTM four-way alignment queue.

## Queue Item

- PR queue item: `<PR #23 / #24 / #25 / #26 / #27 / #28 name>`
- Depends on: `<merged PRs>`

## Summary

- `<what changed>`
- `<what parity risk this closes>`

## Scope

- `<files changed>`
- Historical MLX behavior is unchanged when new controls are absent.
- No WT103/main-machine/result/neo.csv changes.
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
- Are no WT103/main-machine/result/`neo.csv` changes present?
- If WT103 templates are touched, was that separately and explicitly approved?
- Do local tests pass?
- Do GitHub Actions pass?
- Does the PR body report exact verification?

If any answer is no, do not merge until the issue is resolved or explicitly accepted.
