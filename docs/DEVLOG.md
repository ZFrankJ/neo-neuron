# DEVLOG

Durable technical memory for Neo. Keep active queues in `docs/IMPLEMENTATION_PLAN.md`; keep broad priorities in `docs/ROADMAP.md`.

## 2026-07-21 - Manual Compute Accounting Contract

- Decision:
  - Added one shared shape-derived operation-accounting layer for Neo and LSTM;
    backend handling is limited to exposing and normalizing parameter trees.
  - Counted every covered dense operation as exact forward MACs and FLOPs under
    `1 MAC = 2 FLOPs`, with embedding lookup retained as data movement.
  - Kept sigmoid, tanh, other activation, normalization, softmax, loss,
    dropout, bias, and elementwise counts in explicit element categories rather
    than inventing matmul-equivalent costs.
  - Labeled backward dense work and AdamW parameter work as estimates, emitted a
    component/formula coverage manifest, and failed closed on unknown trainable
    parameters, missing coverage, or incompatible parameter shapes.
  - Required Torch and MLX to produce the same logical operation section and
    allowed timing/compute joins only through a separate derived report keyed by
    immutable benchmark, config, checkpoint, and workload identifiers.
  - Bound the stored workload identifier back to the logical workload payload
    during validation, and carried benchmark dry-run status plus provisional
    reasons into every derived timing/compute report.
- Why:
  - Backend profilers and THOP describe framework dispatch or partial operator
    support, not one portable mathematical workload.
  - Algorithmic operation counts must remain distinct from backend/device
    timing, fusion, memory behavior, and hardware utilization.
- Scope:
  - manual compute schema and CLI, parameter coverage audit, derived report,
    deterministic tests, focused Make target, and public workflow documentation
- Impact:
  - Tiny accounting contract checks are safe on the development machine and do
    not execute WT103, load production checkpoints, or change runtime semantics.
  - Formal efficiency execution remains blocked until scaling and comparison
    checkpoint selection are complete.

## 2026-07-21 - Unified Efficiency Benchmark Record Contract

- Decision:
  - Added one backend-neutral benchmark core and CLI with Torch and MLX
    execution adapters for `train_step`, `sequence_eval`, and
    `streaming_decode` workloads.
  - Bound every measured region to explicit backend synchronization, preserved
    every raw nanosecond sample, and made a versioned JSON record the
    authoritative artifact.
  - Required a metadata-bound profile and inferred mapped-versus-backend-native
    checkpoint provenance, content hashes, checkpoint/config compatibility,
    workload scope, runtime identity, parameter breakdown, and telemetry
    capabilities.
  - Defined `train_step` as an isolated full-sequence optimizer update with no
    scheduler and reset recurrent state, and persisted those semantics in every
    record instead of implying the public dataset-driven training trajectory.
  - Captured memory before output validation, retained the maximum per-step
    backend peak, expanded checkpoint checks across model/training semantics,
    and recorded the processor model rather than only the machine architecture.
  - Inferred native-versus-mapped checkpoint provenance, seeded each backend
    before construction and execution, and made dry-run evidence explicitly
    provisional while formal records require complete aligned metadata and an
    exact metadata-bound profile label.
  - Preserved historical MLX Neo `use_checkpoint: true` metadata while recording
    the runtime-inert flag and effective benchmark execution with
    `use_checkpoint: false`; added the focused MLX efficiency suite to macOS CI.
  - Kept raw historical MLX Neo config/checkpoint snapshots intact while
    resolving their omitted `reference_backend` and `rmsnorm_eps` fields to the
    frozen `mlx` and `1e-5` semantics. Each inference is config/checkpoint-hash
    bound; other missing aligned metadata still blocks formal evidence.
  - Kept formal records at a minimum of 20 warm-ups and 100 measurements;
    smaller runs require an explicit `dry_run` label and cannot be silently
    promoted.
- Why:
  - Torch eager execution and MLX lazy execution cannot be compared unless work
    is completed at identical timing boundaries and unsupported telemetry is
    distinguished from a measured zero.
  - Existing THOP estimates, filenames, and backend-native checkpoints do not
    establish wall-clock or cross-backend workload equivalence.
- Scope:
  - shared efficiency schema, validation, persistence, CLI, backend execution
    adapters, deterministic contract tests, and public workflow documentation
- Impact:
  - Tiny dry-run contract checks are safe on a non-main development machine and
    do not change MLX or Torch model/training semantics.
  - Formal efficiency execution remains blocked until LSTM scaling and the
    comparison checkpoint list are frozen; no WT103 process, paper-facing
    benchmark record, historical config, or `neo.csv` row changed.

## 2026-07-14 - 60M-Total / 50M-Recurrent-Core Boundary Diagnostic Contract

- Decision:
  - Added separately named matched and standard-init 10-layer WT103 LSTM
    diagnostic profiles with 60,024,343 trainable parameters on MLX and aligned
    single-bias Torch construction.
  - Kept native MLX initialization in the matched profile by omitting
    initialization keys; confined positive forget bias and orthogonal recurrence
    to the standard-init fallback.
  - Bound both profiles to no LSTM layer dropout, output dropout 0.1,
    no checkpointing, streaming evaluation, a 12-epoch schedule, fresh starts,
    and unique provenance-bearing run tags.
- Why:
  - The historical depth curve confounds added recurrence with repeated
    inter-layer dropout; one controlled geometry isolates that policy before
    any broader scaling work.
  - The epoch-4 gate and fallback order must be reviewable before results exist.
- Scope:
  - two new WT103 LSTM config paths, config-contract tests, verification wiring,
    and user/workflow documentation
- Impact:
  - The profiles are runnable only after a separate explicit start; no WT103
    process, checkpoint, result, historical config, or `neo.csv` row changed.
  - The matched profile remains first, and the standard-init profile remains a
    sequential fallback under the frozen streaming-validation gate.

## 2026-07-14 - Neo MLX Epsilon And Parity Isolation Contract

- Decision:
  - Preserved the frozen Neo MLX RMSNorm epsilon at `1e-5`: a missing value or
    explicit `1e-5` constructs identically, while any other explicit value is
    rejected before model construction.
  - Added tanh to the mapped-weight Neo forward and recurrent-state parity
    matrix across supported recurrent norms and layer counts.
  - Isolated the MLX process-global default device around every Neo parity test.
- Why:
  - The MLX adapter previously recorded arbitrary explicit Neo epsilon values
    without executing them, tanh was absent from the durable matrix, and the
    public-loop test could leave later tests on CPU depending on module order.
- Scope:
  - `src/runtime/backends/mlx_backend.py`
  - `tests/test_mlx_reference_parity.py`
  - public error-contract documentation
- Impact:
  - Existing Neo MLX runs with a missing epsilon or `1e-5` keep unchanged
    runtime semantics and numerical behavior.
  - Unsupported explicit values now fail clearly, tanh parity is enforced, and
    focused MLX parity tests no longer depend on execution order.

## 2026-07-14 - WT103 Revalidation Policy Activated

- Decision:
  - Activated a staged WT103 revalidation policy after the four-path backend and
    model-fairness audit.
  - Preserved every historical WT103 config, run tag, checkpoint, and result
    label; new experiments must use separately named profiles.
  - Chose one `d_model=790`, `n_layers=10`, approximately 60M-total / 50M-recurrent-core
    matched-no-layer-dropout MLX LSTM as the first boundary diagnostic, with a
    standard-init fallback and an epoch-4 validation review under a 12-epoch
    scheduler contract.
  - Froze `83.54` as the epoch-4 streaming-validation continuation threshold
    before the new result is observed; test PPL remains excluded from profile
    and checkpoint selection.
  - Classified that threshold as an operational resource gate rather than a
    statistical claim; paper-facing recovery evidence still requires a separate
    repeated-seed variance packet.
  - Excluded clean MLX Neo retraining when explicit metadata fields do not change
    training mathematics; existing checkpoints remain eligible for streaming
    reevaluation.
- Why:
  - Configuration alone does not close silent Neo MLX epsilon handling, tanh
    parity coverage, or test-device isolation gaps.
  - The historical LSTM depth curve is parameter-budget matched but includes
    repeated inter-layer dropout, so a single controlled diagnostic is more
    informative than immediately rebuilding every scaling point.
- Scope:
  - planning, progress, training-policy, and roadmap documentation
- Impact:
  - Implementation is split into a contract/test-hardening PR and a dependent
    WT103 diagnostic-config PR.
  - No WT103 run is started by this decision, and full LSTM scaling remains
    conditional on the diagnostic result.

## 2026-07-14 - Aligned LSTM Trial Profile And Queue Closure

- Decision:
  - Added `configs/alignment/lstm_standard_init_trial.yaml` as a small,
    provenance-explicit Wikitext-2 readiness profile using MLX-reference
    scheduler timing, single effective bias, `rmsnorm_eps: 1e-5`, orthogonal
    recurrence, positive forget bias, no LSTM layer dropout, no checkpointing,
    streaming evaluation, and a 10%-of-epoch warmup followed by cosine decay to
    `min_lr`.
  - Fixed the reporting label as `standard-init no-layer-dropout RMSNorm-LSTM`
    and recorded an equal Torch/MLX trainable count of 3,546,833.
  - Closed the local LSTM correction queue after the profile contract and all
    four-way parity gates were represented in checked-in tests and docs.
- Why:
  - Result work needs one explicit, testable profile instead of reconstructing
    alignment controls from historical backend defaults.
  - A readiness fixture can close implementation ambiguity without authorizing
    WT103 training or silently relabeling prior artifacts.
- Scope:
  - `configs/alignment/lstm_standard_init_trial.yaml`
  - `tests/test_lstm_alignment_profile.py`
  - `Makefile`
  - user, training, roadmap, progress, and queue documentation
- Impact:
  - `legacy MLX LSTM`, `matched no-layer-dropout RMSNorm-LSTM`, and
    `standard-init no-layer-dropout RMSNorm-LSTM` now have distinct durable
    labels.
  - Existing WT103 configs, run tags, historical artifacts, MLX runtime
    semantics, production machines, and `neo.csv` remain unchanged.
  - The next step requires a separately approved result-production and
    variance plan.

## 2026-07-14 - MLX-Reference Scheduler Timing

- Decision:
  - Shifted the Torch cosine/warmup schedule by one update only when
    `reference_backend: mlx`, so Torch update one uses MLX schedule step one.
  - Kept native Torch configs on the historical step-zero start.
  - Added direct per-update learning-rate coverage and a deterministic public
    LSTM loop using streaming batches, cosine warmup, and explicit aligned
    bias, RMSNorm, dropout, and initialization controls.
- Why:
  - MLX assigns `scheduler.lr(global_step + 1)` before each optimizer update,
    while Torch `LambdaLR` initialized the first update at step zero and only
    advanced after the optimizer update.
- Scope:
  - `src/train/schedulers.py`
  - `tests/test_scheduler_timing.py`
  - `tests/test_lstm_training_parity.py`
  - `Makefile`
  - training and workflow documentation
- Impact:
  - Explicit MLX-reference Torch runs now match MLX per-update learning rates
    and deterministic public-loop LSTM metrics.
  - Historical native Torch schedules, MLX runtime semantics, WT103 configs,
    runs, results, and `neo.csv` remain unchanged.

## 2026-07-14 - Cross-Backend LSTM Baseline Controls

- Decision:
  - Added MLX support for explicit `lstm_layer_dropout`, `forget_bias_init`,
    and `recurrent_init` controls already available on Torch.
  - Defined the opt-in standard initialization contract as Xavier input
    matrices, `xavier_uniform` or gate-wise `orthogonal` recurrent matrices,
    zero non-forget biases, and the configured effective forget bias.
  - Kept key presence as the MLX compatibility boundary: missing init keys
    retain native MLX uniform weights and random bias values exactly, while
    `lstm_layer_dropout` alone does not reinitialize parameters.
- Why:
  - Matched and stronger LSTM profiles must be constructible on the MLX result
    backend without silently changing historical MLX runs.
  - Equal seeded tensors are not portable across frameworks, but independent
    initialization invariants and explicit dropout policy are testable.
- Scope:
  - `src/runtime/backends/mlx_backend.py`
  - `tests/test_lstm_init_controls.py`
  - `Makefile`
  - user and workflow documentation
- Impact:
  - Torch and MLX can now construct explicit matched no-layer-dropout and
    standard-init LSTM profiles.
  - Historical configs, WT103 templates, runs, checkpoints, results, and
    `neo.csv` remain unchanged.

## 2026-07-14 - LSTM Gradient And Trajectory Parity

- Decision:
  - Added deterministic MLX-reference LSTM training parity for the explicit
    single-bias, `rmsnorm_eps: 1e-5`, no-dropout, fixed-batch,
    no-checkpoint profile.
  - Set `1e-6` loss and `1e-5` gradient/update/12-step trajectory tolerances,
    with recurrent-state `rtol=1e-5` and `atol=1e-6`, from measured local
    maxima of `7.16e-7` loss drift, `5.93e-7` gradient-norm drift, `2.99e-8`
    parameter drift, and `1.20e-7` recurrent state drift without weakening the
    existing Neo contract.
  - Reset Torch and MLX initialization to fixed seed `20260714` before every
    training-parity test so the measured envelope is reproducible across fresh
    processes.
  - Kept optimizer state backend-native: same-backend resume is exact, while a
    cross-backend load with optimizer state now warns and leaves the
    destination optimizer state untouched.
- Why:
  - Forward parity alone did not prove that every mapped gradient, decay role,
    Adam update, recurrent state, and short training trajectory matched MLX.
  - The historical split-bias Torch path doubles the effective gate-bias update;
    the new regression proves it falls outside the aligned update envelope.
- Scope:
  - `tests/test_lstm_training_parity.py`
  - `src/train/checkpointing.py`
  - `src/runtime/backends/mlx_backend.py`
  - `Makefile`
  - user and workflow documentation
- Impact:
  - `make lstm-parity` now covers deterministic LSTM inference, gradients,
    public optimizer roles, one update, short trajectory, and optimizer resume.
  - Random-batch and dropout-mask replay, cross-backend optimizer-state mapping,
    WT103 runs, results, and `neo.csv` remain outside this contract.

## 2026-07-14 - LSTM Forward And Checkpoint Parity

- Decision:
  - Added explicit LSTM `rmsnorm_eps` handling with `1e-5` as the aligned
    Torch/MLX value.
  - Preserved the historical Torch dtype-derived RMSNorm epsilon when the field
    is absent; MLX accepts missing or explicit `1e-5` and rejects other
    explicit values without changing its native runtime semantics.
  - Added LSTM checkpoint guards for bias mode, recurrent norm, norm placement,
    and RMSNorm epsilon, with missing fields treated as legacy/provisional.
- Why:
  - PyTorch RMSNorm defaults to a dtype-derived epsilon while MLX uses `1e-5`,
    which caused mapped-weight LSTM logits to diverge despite matching weights.
  - Forward and checkpoint claims need deterministic coverage across supported
    norm modes and layer counts before training-trajectory parity is meaningful.
- Scope:
  - `src/models/lstm_lm.py`
  - `src/runtime/backends/torch_backend.py`
  - `src/runtime/backends/mlx_backend.py`
  - `src/runtime/checkpoint_compat.py`
  - `tests/test_lstm_forward_parity.py`
  - `Makefile`
  - `justfile`
  - `.github/workflows/tests.yml`
  - user and workflow documentation
- Impact:
  - Same mapped weights now match forward logits, recurrent state, and loss for
    one/two-layer LSTMs using no norm, LayerNorm, or RMSNorm.
  - MLX-to-Torch and Torch-to-MLX checkpoint conversion preserves evaluation
    loss under the aligned profile.
  - Historical Torch configs and checkpoints are not silently relabeled as
    aligned; no WT103 run, result, or `neo.csv` behavior changes.

## 2026-07-14 - LSTM Effective-Bias Contract

- Decision:
  - Added `lstm_bias_mode: split | single`; missing Torch config keeps the historical two-trainable-bias `split` path, while MLX remains natively `single`.
  - Explicit Torch `single` mode keeps `bias_hh` in checkpoint state for compatibility but freezes it, leaving exactly one trainable effective gate bias per layer.
  - Torch-to-MLX conversion continues to sum retained split-bias state and warns that cross-backend optimizer resume is not equivalent.
- Why:
  - Torch `bias_ih` and `bias_hh` each receive the MLX-equivalent gradient, so updating both doubles the effective bias update even when mapped-weight forward outputs match.
  - Retaining the frozen compatibility tensor preserves evaluation of historical Torch checkpoints without changing MLX runtime semantics.
- Scope:
  - `src/models/lstm_lm.py`
  - `src/runtime/backends/torch_backend.py`
  - `src/runtime/backends/mlx_backend.py`
  - `src/runtime/checkpoint_compat.py`
  - `tests/test_lstm_bias_contract.py`
  - `docs/training.md`
- Impact:
  - Explicit single-bias Torch LSTM models match MLX trainable parameter counts and mapped effective-bias gradients.
  - Historical Torch configs remain split-bias models; historical MLX configs and initialization remain unchanged.
  - No Neo, Transformer, WT103 config, run, result, or `neo.csv` behavior changes.

## 2026-07-11 - Cross-Backend LSTM Optimizer Grouping

- Decision:
  - The MLX-reference Torch optimizer recognizes both Torch `lstm.*` and MLX-style `lstm_layers.*` recurrent parameter namespaces.
  - Torch LSTM pre-norms remain in the zero-decay norm bucket rather than inheriting recurrent weight decay.
- Why:
  - Torch LSTM recurrent weights were receiving projection decay when `reference_backend: mlx` because the classifier only recognized MLX parameter names.
  - Cross-backend comparison requires grouping by parameter role while preserving each backend's native names.
- Scope:
  - `src/train/optim.py`
  - `tests/test_optimizer_grouping.py`
  - `docs/training.md`
- Impact:
  - MLX-reference Torch LSTM runs now apply configured embedding, projection, recurrent, and zero-decay buckets consistently.
  - Neo and Transformer grouping behavior is unchanged and covered by regression tests; MLX runtime semantics are unchanged.

## 2026-07-11 - GPT-2-Style Transformer Control

- Decision:
  - Added an opt-in `transformer_variant: gpt2` with pre-norm GELU blocks, optimized causal attention, 0.02 normal initialization, and depth-scaled residual projections.
  - Kept missing `transformer_variant` mapped to `legacy` so old configs and checkpoints retain their historical interpretation.
  - Reject checkpoint loads when the recorded Transformer variant conflicts with the requested variant; missing historical metadata is interpreted as `legacy`.
- Why:
  - The hand-rolled Transformer was useful as an internal smoke control but was too weakly specified for paper-facing Transformer comparisons.
  - Torch and MLX need one explicit architecture and checkpoint contract rather than similarly named but behaviorally divergent controls.
- Scope:
  - `src/models/transformer_lm.py`
  - `src/runtime/backends/torch_backend.py`
  - `src/runtime/backends/mlx_backend.py`
  - `src/runtime/checkpoint_compat.py`
  - `configs/wt103/transformer_30m.yaml`
  - `tests/test_gpt2_transformer_control.py`
  - `README.md`
  - `docs/training.md`
- Impact:
  - The checked-in WT103 Transformer template is clearly labeled GPT-2-style and maps checkpoints across Torch and MLX with tested causal forward parity.
  - Historical Transformer runs remain lightweight internal controls and are not relabeled.
  - No WT103 training, dataset download, checkpoint artifact, or `neo.csv` change was performed.

## 2026-07-11 - Config Labels And Neo Activation Provenance

- Decision:
  - Kept WT2 config filenames as compatibility paths while replacing their numeric implications in reporting guidance with small/large labels and exact parameter counts.
  - Set future WT103 Neo config templates to `activation_id: tanh` and added `tanh` to their run tags.
- Why:
  - WT2 `6m` and `25m` filenames materially understate their current parameter counts, while Neo result interpretation requires a visible boundary between new tanh runs and historical custom-activation runs.
- Scope:
  - `configs/wt103/neo_20m.yaml`
  - `configs/wt103/neo_30m.yaml`
  - `configs/wt103/neo_50m.yaml`
  - `tests/test_config_activation_provenance.py`
  - `README.md`
  - `docs/training.md`
- Impact:
  - Future WT103 Neo runs launched from checked-in configs are explicitly tanh runs.
  - Historical `id4` and `id5` checkpoints and saved config snapshots retain their original provenance and are not relabeled.
  - No WT103 training, dataset download, checkpoint mutation, or `neo.csv` change was performed.

## 2026-07-11 - Explicit Recurrent Evaluation Regimes

- Decision:
  - Added `eval_regime: block_reset | streaming` across Torch and MLX evaluation, with `block_reset` as the compatibility default.
  - Defined streaming evaluation as state carry along contiguous batch lanes, never across lanes, and exposed the selected regime in evaluation metrics and CLI output.
- Why:
  - Training can carry recurrent state while historical evaluation reset it per block, so paper-facing perplexity needs an explicit state-lifetime contract.
  - Existing configs and checkpoints must retain their historical block-reset interpretation unless a run deliberately opts into streaming evaluation.
- Scope:
  - `src/runtime/eval_semantics.py`
  - `src/train/eval.py`
  - `src/runtime/backends/mlx_backend.py`
  - `scripts/eval.py`
  - `tests/test_recurrent_eval_semantics.py`
  - `README.md`
  - `docs/training.md`
- Impact:
  - Future result tables must name the evaluation regime.
  - Streaming evaluation is deterministic and backend-aligned without changing MLX model semantics or historical result provenance.

## 2026-07-11 - Opt-In Standard-Init LSTM Control

- Decision:
  - Added opt-in PyTorch LSTM controls for positive forget-gate bias, gate-wise orthogonal recurrent initialization, and independent inter-layer dropout.
  - Preserved zero forget bias, Xavier-uniform recurrence, and `dropout`-as-layer-dropout as compatibility defaults.
- Why:
  - A stronger LSTM comparator should follow common LSTM initialization practice without silently changing historical control configs or implying one-to-one Neo equivalents.
  - Reusing output dropout between LSTM layers must be an explicit policy choice for future result labels.
- Scope:
  - `src/models/lstm_lm.py`
  - `src/runtime/backends/torch_backend.py`
  - `tests/test_lstm_init_controls.py`
  - `README.md`
  - `docs/training.md`
- Impact:
  - Future PyTorch runs can be labeled `standard-init RMSNorm-LSTM` with `forget_bias_init: 1.0`, `recurrent_init: orthogonal`, and explicit `lstm_layer_dropout`.
  - Historical runs remain `RMSNorm-LSTM matched control`; MLX runtime semantics and existing checkpoint interpretation are unchanged.

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
