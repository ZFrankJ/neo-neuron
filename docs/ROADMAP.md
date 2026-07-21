# Roadmap

This roadmap lists broad project priorities only. Checkpoint state belongs in `docs/PROGRESS.md`; strict task and PR execution belongs in `docs/IMPLEMENTATION_PLAN.md`.

## Priority Order

## 1. Backend Parity Against MLX

- Status: Neo and LSTM alignment complete locally; external CUDA validation remains
- Why now:
  - MLX has produced the trusted result line, while older PyTorch MPS runs showed numerical and memory contamination risk.
  - PyTorch is still needed as the bridge toward CUDA, but it must prove parity against MLX in controlled steps first.
- Focus:
  - Preserve MLX as the frozen reference.
  - Keep PyTorch CPU as the semantic bridge.
  - Preserve the completed same-weight LSTM parity contract when future result
    profiles or backends are added.
  - Keep PyTorch MPS no-checkpoint parity as the trusted local PyTorch path.
  - Keep CUDA validation gated by an explicit Nvidia GPU preflight.
- Exit condition:
  - The Neo MPS parity exit and LSTM four-way alignment exit in
    `docs/IMPLEMENTATION_PLAN.md` are satisfied, CUDA preparation is skip-safe
    on non-CUDA machines, and reproducers know to run the CUDA preflight before
    CUDA claims.
- Expanded plan:
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`

## 2. CUDA Parity And Runner Enablement

- Status: no active queue; blocked on external Nvidia GPU access
- Why now:
  - The long-term target is PyTorch CUDA parity against MLX, but this repo does not currently have local CUDA hardware and standard GitHub-hosted runners for individual repos are not Nvidia GPU runners.
- Focus:
  - Reuse the PyTorch CPU bridge and parity audit harness.
  - Keep CUDA tests full precision, no checkpoint, no compile, and no fused optimizer until baseline parity is proven.
  - Do not queue Nvidia GitHub CI while no real GPU runner exists.
  - Require reproducers to confirm `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py` runs instead of skips before making CUDA claims.
- Exit condition:
  - CUDA parity tests run on an Nvidia device and pass the same small-scale parity envelope used for MPS. Until then, the active PR queue stays empty.
- Expanded plan:
  - `docs/IMPLEMENTATION_PLAN.md`

## 3. Baseline Alignment And Result Readiness

- Status: local alignment complete; WT103 result preparation moved to priority 4
- Why now:
  - Neo backend parity is locally complete, but paper-facing claims also need
    defensible baselines, explicit evaluation semantics, LSTM parity, and
    result provenance.
  - The historical LSTM control is a normalized recurrent baseline, not a
    vanilla LSTM baseline; that is acceptable only when named and tested as
    such.
  - The completed four-way correction keeps historical backend-native LSTM
    profiles intact while providing explicit matched and standard-init aligned
    profiles.
  - Neo is moving toward a tanh activation framing, so future result labels must separate cell-structure claims from older custom-activation runs.
- Focus:
  - Preserve the strengthened LSTM baseline without changing old result
    interpretation.
  - Use the proven MLX/Torch LSTM forward, gradient, optimizer, trajectory,
    checkpoint, and public-loop contracts for future approved result work.
  - Keep recurrent block-reset versus streaming-state evaluation explicit and
    report the selected regime with each result.
  - Keep WT2 labels honest by using small/large names plus exact parameter counts.
  - Preserve the GPT-2-style Transformer control before making Transformer
    comparison claims.
  - Preserve optimizer grouping guards that prevent silent Torch/MLX parity or
    LSTM comparison drift.
- Exit condition:
  - The LSTM correction queue passes its parity gates, historical profiles stay
    provenance-bound, and a separate approved result-production plan selects
    the aligned profile before paper-quality runs are scheduled.
- Expanded plan:
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`

## 4. Result Production And WT103 Revalidation

- Status: corrected LSTM scaling active; 50M-total anchor passed its epoch-4 gate
- Why now:
  - Old PyTorch MPS WT103 rows are not trusted, and new result production should wait until baseline alignment is explicit.
  - The four-path audit found that the historical MLX results remain valid for
    their recorded profiles, but backend overrides and the LSTM depth comparison
    still have explicit configuration and regularization confounds.
- Focus:
  - Preserve historical WT103 configs, run tags, checkpoints, and result labels.
  - Close the remaining Neo MLX contract and parity-test gaps before adding new
    WT103 profiles.
  - Preserve the separately named 60M-total / 50M-recurrent-core
    matched-no-layer-dropout and standard-init LSTM boundary profiles with
    config-contract tests.
  - Do not retrain clean MLX Neo checkpoints when the new explicit fields do not
    change their training mathematics; use streaming reevaluation instead.
  - Complete the user-approved matched-no-layer-dropout 50M-total /
    40M-recurrent-core LSTM anchor under its 12-epoch schedule before selecting
    deeper corrected-profile scaling points.
  - Keep unified Torch/MLX timing and backend-neutral manual compute measurement
    blocked until scaling and checkpoint selection are complete.
- Exit condition:
  - The new profiles are merged without repurposing historical paths, the
    corrected LSTM scaling profile is classified on completed runs, deeper
    retraining is either justified or explicitly rejected, and selected clean
    Neo checkpoints remain provenance-bound.
- Expanded plan:
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`
