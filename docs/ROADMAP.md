# Roadmap

This roadmap lists broad project priorities only. Checkpoint state belongs in `docs/PROGRESS.md`; strict task and PR execution belongs in `docs/IMPLEMENTATION_PLAN.md`.

## Priority Order

## 1. Backend Parity Against MLX

- Status: Neo locally complete; LSTM alignment correction active; CUDA validation blocked on Nvidia hardware access
- Why now:
  - MLX has produced the trusted result line, while older PyTorch MPS runs showed numerical and memory contamination risk.
  - PyTorch is still needed as the bridge toward CUDA, but it must prove parity against MLX in controlled steps first.
- Focus:
  - Preserve MLX as the frozen reference.
  - Keep PyTorch CPU as the semantic bridge.
  - Extend the same-weight parity contract to LSTM before treating Torch LSTM as
    an MLX reproduction path.
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

- Status: active LSTM correction queue
- Why now:
  - Neo backend parity is locally complete, but paper-facing claims also need
    defensible baselines, explicit evaluation semantics, LSTM parity, and
    result provenance.
  - The current LSTM control is a normalized recurrent baseline, not a vanilla LSTM baseline; that is acceptable only if named and tested as such.
  - A four-way implementation audit found that the active MLX LSTM has
    LSTM-only inter-layer dropout and native initialization, while Torch LSTM
    still differs in effective-bias training and RMSNorm epsilon.
  - Neo is moving toward a tanh activation framing, so future result labels must separate cell-structure claims from older custom-activation runs.
- Focus:
  - Strengthen the LSTM baseline without changing old result interpretation.
  - Prove MLX/Torch LSTM forward, gradient, optimizer, trajectory, and
    checkpoint parity under an explicit aligned profile.
  - Decide whether recurrent evaluation should use block-reset or streaming-state semantics, and report both if needed.
  - Keep WT2 labels honest by using small/large names plus exact parameter counts.
  - Upgrade the Transformer control toward a GPT-2-style internal baseline before making Transformer comparison claims.
  - Fix optimizer grouping edge cases that could silently make Torch/MLX parity or LSTM comparisons misleading.
- Exit condition:
  - The LSTM correction queue passes its parity gates, historical profiles stay
    provenance-bound, and a separate approved result-production plan selects
    the aligned profile before paper-quality runs are scheduled.
- Expanded plan:
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`

## 4. Result Production And WT103 Revalidation

- Status: later
- Why now:
  - Old PyTorch MPS WT103 rows are not trusted, and new result production should wait until baseline alignment is explicit.
- Focus:
  - Do not rerun WT103 or rehabilitate old PyTorch result rows until the parity ladder and baseline-alignment queue are complete.
- Exit condition:
  - A separate approved plan defines hardware, configs, result labels, and acceptance criteria.
- Expanded plan:
  - not active
