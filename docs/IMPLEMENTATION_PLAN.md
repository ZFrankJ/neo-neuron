# Implementation Plan

Strict execution contract for Neo backend parity, baseline alignment, WT103
revalidation, and deferred efficiency measurement.

Previous tracking issue: https://github.com/ZFrankJ/neo-neuron/issues/2

Issue #2 is closed. PR #28 completed the local LSTM four-way correction queue;
open a new tracking issue only when explicitly requested.

## Current State

```text
completed packet PR 1: https://github.com/ZFrankJ/neo-neuron/pull/29
completed packet PR 2: https://github.com/ZFrankJ/neo-neuron/pull/30
completed packet PR 3: https://github.com/ZFrankJ/neo-neuron/pull/31
active packet PR 4: manual compute accounting on codex/feat/manual-compute-accounting
active experiment: 50M-total / 40M-recurrent-core matched-no-layer-dropout LSTM anchor
blocked follow-up: formal efficiency execution after LSTM scaling and checkpoint selection
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

The user-approved result sequence now starts by rebuilding the completed
`d_model=790`, `n_layers=8` LSTM anchor under the matched-no-layer-dropout
profile. This gives an exact historical-LSTM control and a near-equal-total-
parameter Neo comparison before deeper LSTM points are selected. Efficiency
measurement is planned now but must not run concurrently with, modify, or
select checkpoints for the active scaling sequence.

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
- The first corrected-profile scaling anchor is the `d_model=790`, `n_layers=8`
  MLX LSTM: 50,030,843 total / approximately 40M recurrent-core parameters,
  configured for 12 epochs and reviewed after epoch 4.
- Deeper LSTM scaling is authorized only while the selected profile remains
  healthy under predeclared streaming-validation gates; clean MLX Neo
  checkpoints are reevaluated rather than retrained.
- Unified Torch/MLX efficiency benchmarking remains blocked until the LSTM scaling
  profile and comparison checkpoints are frozen.

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
- PR #30: https://github.com/ZFrankJ/neo-neuron/pull/30
  - Added test-covered 60M-total / 50M-recurrent-core matched and standard-init
    boundary diagnostic profiles without changing historical WT103 paths.
- PR #31: https://github.com/ZFrankJ/neo-neuron/pull/31
  - Added the unified Torch/MLX wall-clock and memory harness, immutable
    benchmark records, explicit workload semantics, and tiny dry-run contract
    verification without starting formal benchmark execution.

## Active Execution Queue

Execute exactly one packet at a time in this order.

### Corrected LSTM Scaling - Active Experiment, Not A Code PR

- The active run is the matched-no-layer-dropout `d_model=790`, `n_layers=8`
  MLX LSTM with 50,030,843 total / approximately 40M recurrent-core parameters.
- Keep `epochs: 12`; do not shorten the cosine schedule for an early review.
- Use streaming validation PPL for checkpoint and continuation decisions. Do
  not inspect test PPL until a profile and checkpoint have been selected.
- The frozen historical same-geometry epoch-4 streaming-validation baseline in
  `experiments/wt103/eval_regimes_epoch4_20260714.csv` is `82.57`.
- The corrected run passed its epoch-4 operational gate at `79.92` and continues
  to epoch 12. This is one-seed resource-allocation evidence, not a
  paper-facing significance claim.
- Any NaN or infinity is an optimization failure. A checkpoint or data failure
  invalidates the affected run and must be repaired before interpretation.
- Do not run another production model or any efficiency benchmark concurrently
  with the active scaling run.
- After completion, select the checkpoint by streaming validation, evaluate the
  selected checkpoint on streaming test once, record the result under its exact
  profile and parameter count, and predeclare the next deeper scaling point.
- Clean Neo checkpoints remain authoritative and are reevaluated rather than
  retrained. Historical LSTM checkpoints remain historical-profile evidence and
  must not be mixed into a corrected-profile scaling fit.

### Unified Torch/MLX Wall-Clock And Memory Harness - Completed Code Packet

Completed in PR #31 after development was explicitly approved on 2026-07-21 on a non-main machine while
the LSTM scaling run continues. Tiny deterministic dry runs are permitted for
contract verification. Formal benchmark execution and paper-facing records
still depend on completion of the scaling sequence and a frozen list of Neo and
LSTM comparison checkpoints. This first efficiency-measurement PR must not
touch or compete with the main training machine.

Public contract:

- Add one backend-neutral benchmark CLI, provisionally
  `scripts/benchmark_efficiency.py`, and one versioned result schema without
  changing training or model runtime semantics.
- Support `--backend mlx|torch` and an explicit device. Resolve model
  construction, checkpoint loading, and execution through the existing backend
  contracts; do not duplicate model mathematics in the benchmark.
- Keep one shared benchmark core for workload setup, warm-up, sample capture,
  aggregation, persistence, and validation. Backend adapters own only execution,
  synchronization, and available memory telemetry.
- Accept an explicit config, checkpoint, backend, device, workload, batch size,
  sequence length, warm-up count, measured-iteration count, repetition
  identifier, and output path. Never infer a paper-facing profile from a
  filename alone.
- Support separately labeled `train_step`, `sequence_eval`, and
  `streaming_decode` workloads. Define `train_step` as an isolated optimizer
  update with fresh-then-warmed optimizer state, no scheduler, reset recurrent
  state, and full-sequence backpropagation; reject a shorter configured TBPTT
  contract instead of silently changing it. Model-only timing and end-to-end
  loop timing are distinct scopes and must never be combined into one number.
- Preallocate deterministic token inputs outside the measured region. Exclude
  dataset download, tokenization, checkpoint loading, compilation/warm-up, and
  result serialization from model-only timing.
- Use `time.perf_counter_ns()` with backend synchronization immediately before
  and after every measured region: `mx.synchronize()` for MLX, eager completion
  for Torch CPU, `torch.mps.synchronize()` for MPS, and
  `torch.cuda.synchronize(device)` for CUDA. Lazy or asynchronous work must
  never cross a timing boundary.
- Seed each backend before model construction and again before execution.
  Record that RNG streams are backend-local; equal seeds prove within-backend
  repeatability and do not imply identical Torch/MLX dropout masks.
- Fail clearly when the requested backend/device or required synchronization
  primitive is unavailable. MPS and CUDA support remains skip-safe and does not
  weaken their existing scientific-acceptance gates.
- Run at least 20 unreported warm-up iterations and 100 measured iterations per
  process unless an approved dry run establishes a larger minimum for stable
  intervals. Preserve every raw sample.
- Reset backend peak-memory counters after warm-up and between measurements when
  a supported API exists. Capture measured-work memory before output validation
  can allocate temporary buffers, and retain the maximum measured backend peak.
  Report process RSS for every backend plus backend active/peak allocation,
  tokens per second, milliseconds per token, and milliseconds per step. An
  unsupported memory field must be recorded as unavailable, never as zero.
- Emit a versioned JSON record containing raw samples and summary statistics,
  including median, quartiles, and 10th/90th percentiles. Any CSV table is a
  derived view, not the authoritative record.
- Record git commit, config snapshot, checkpoint identity and
  execution-semantic metadata, exact trainable parameter count and breakdown,
  model/profile labels, activation and norm settings, backend, device, dtype,
  framework/Python/OS versions, hardware identifier including the processor
  model, workload dimensions, seed, synchronization policy, telemetry
  capability flags, and whether data handling is included.
- Preserve historical config separately from effective benchmark execution.
  The only `use_checkpoint: true` exception is a native MLX Neo checkpoint,
  where the historically recorded flag had no MLX runtime effect; record the
  exact override to benchmark execution with `use_checkpoint: false`.
  For the same frozen historical MLX Neo contract only, resolve omitted
  `reference_backend` and `rmsnorm_eps` metadata to `mlx` and `1e-5` while
  preserving both raw snapshots. Bind each inference to the config/checkpoint
  hashes and record the frozen-semantics reason; do not relax any other missing
  metadata gate.
- Require equivalent logical workload definitions across Torch and MLX. A
  cross-backend comparison must record whether weights were mapped from the
  same checkpoint or loaded natively. Infer that provenance from checkpoint and
  execution backends rather than trusting a caller-supplied assertion.
- Formal records require complete model-family aligned checkpoint metadata and
  an exact profile label bound to config and checkpoint metadata. Every dry run
  is explicitly provisional and records missing metadata or label reasons.
- Reject metadata mismatch, unsupported backends or workloads, non-positive
  iteration counts, non-finite outputs, and attempts to overwrite an existing
  authoritative record without an explicit replacement flag.
- Do not depend on THOP or another generic operation profiler. Existing THOP
  fields retain historical provenance but are excluded from formal efficiency
  claims.

Test-first and verification contract:

- First tests freeze the shared CLI/schema, adapter boundary, synchronization
  order, warm-up exclusion, raw-sample persistence, percentile aggregation,
  overwrite guard, capability encoding, metadata mismatch failures,
  backend-local RNG repeatability, formal/provisional evidence classification,
  inferred checkpoint provenance, a formal fixture derived from the checked-in
  legacy WT103 Neo config shape, historical MLX Neo execution overrides, and
  tiny deterministic fixtures and a controllable clock.
- A required tiny Torch CPU integration test and a skip-safe tiny MLX integration
  test must emit the same schema and prove that measured work is completed at
  the timing boundary. MPS and CUDA integration tests remain opt-in and
  skip-safe.
- A mapped-weight tiny workload must verify that Torch and MLX benchmark the
  same input/output shapes and semantic workload before cross-backend timing is
  accepted. CI must not enforce hardware performance thresholds.
- Run the focused benchmark tests, `make torch-validate`, `make mlx-parity`,
  `make check`, and `git diff --check`. The PR itself must not run WT103 or
  produce paper-facing benchmark records.

### Manual Compute Accounting - Active Code Packet

Depends on the unified timing harness contract. Keep arithmetic accounting
independent from measured wall-clock performance so hardware utilization is not
misreported as algorithmic compute.

Public contract:

- Add one backend-neutral, shape-derived operation counter for Neo and LSTM that
  walks supported logical model components and fails closed on unknown trainable
  modules. Do not maintain independent Torch and MLX formulas for the same
  mathematical operation.
- Count matrix multiply-accumulate operations exactly from tensor shapes and
  report both MACs and FLOPs under the explicit convention `1 MAC = 2 FLOPs`.
- Include recurrent matrices, input/output projections, tied or untied output
  heads, and every other parameterized dense operation executed by the selected
  workload. Treat embedding lookup as data movement rather than a dense MAC.
- Report sigmoid, tanh, activation, normalization, softmax, loss, optimizer, and
  other scalar/elementwise operation counts by category instead of silently
  assigning them an arbitrary matmul-equivalent cost.
- Label forward compute as exact under the declared formula coverage. Label
  backward and optimizer compute as estimates unless every derivative and
  update formula is explicitly enumerated and tested.
- Treat audited operations as the mathematical workload, independent of kernel
  fusion, layout, framework dispatch, or hardware utilization. Those effects
  belong only to measured backend/device timing.
- Emit a coverage manifest mapping every supported model component to its
  formula and reject incomplete coverage. Equivalent Torch and MLX model
  structures must produce identical operation records for the same workload;
  disagreement is a contract failure. Do not use THOP output as an oracle.
- Merge the audited operation record with timing results only through a derived
  report keyed by immutable config/checkpoint/workload identifiers.

Test-first and verification contract:

- First tests use hand-computable tiny Neo and LSTM shapes, tied/untied heads,
  supported normalization and activation settings, and deliberate unknown
  modules to prove fail-closed coverage.
- Construct equivalent tiny Torch and MLX models and require identical logical
  operation counts. Cross-check parameterized operation coverage against both
  model parameter trees; parameter count agreement alone is not accepted as
  compute agreement.
- Run focused compute-accounting tests, `make torch-validate`,
  `make mlx-parity`, `make check`, and `git diff --check`.
- `make efficiency-check` runs the focused Torch/MLX harness integrations and is
  required in the macOS MLX GitHub job.

### Formal Efficiency Matrix - Blocked Experiment, Not A Code PR

Run only after corrected LSTM scaling, both code packets, and their verification
are complete:

- Freeze near-equal-total-parameter Neo/LSTM checkpoint pairs before measuring.
  Parameter-and-token parity remains the primary architecture comparison;
  compute and wall-clock are separately labeled secondary comparisons.
- Use the same workload inputs, dtype, batch/sequence dimensions, logical
  operation record, and benchmark commit for every cross-backend pair. Record
  backend, device, synchronization, and telemetry capabilities explicitly.
- Separate two matrices: architecture comparison holds backend/device fixed
  while varying Neo versus LSTM; backend comparison holds model, weights, and
  workload fixed while varying Torch versus MLX. Never vary architecture and
  backend in the same unlabeled comparison.
- Run with no production training or unrelated accelerator workload active.
  Use at least five fresh-process repetitions and alternate model and backend
  order across repetitions to reduce cache, order, and thermal bias.
- Measure training at `batch_size=20`, `block_size=256`; measure streaming decode
  at batch sizes 1 and 20. Any additional workload is a separately named row.
- Publish raw JSON records, median and interval summaries, exact parameters,
  audited MACs/FLOPs, throughput, latency, and peak memory. Preserve failed or
  excluded repetitions with explicit reasons.
- Do not infer PPL-versus-training-time from microbenchmark throughput or file
  modification times. That claim requires monotonic elapsed time recorded
  during a separately approved timed training run.
- Do not generalize wall-clock results across backends, devices, or hardware.
  Torch MPS measurements remain systems diagnostics while MPS is outside
  scientific result production, and CUDA measurements require a real validated
  Nvidia environment. The portable claim is limited to audited operation
  counts; measured latency and throughput remain
  machine/backend/device-specific.

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

- Formal unified Torch/MLX efficiency benchmark execution before corrected LSTM
  scaling and checkpoint selection are complete; only tiny explicitly labeled
  dry-run contract checks are permitted during harness development.
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
- Using legacy THOP GFLOPs fields, microbenchmark extrapolation, or filesystem
  timestamps as formal compute- or wall-clock-parity evidence.
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
