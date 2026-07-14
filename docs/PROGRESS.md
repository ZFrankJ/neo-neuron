# Progress

Checkpoint tracker for the active large goal. This file sits between `docs/ROADMAP.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Active Large Goal

Roadmap priority:

`3. Baseline Alignment And Result Readiness`

Current position:

- Neo backend parity is locally complete, with CUDA validation still blocked outside the local queue
- the first paper-facing baseline slice is merged and provides opt-in standard-init PyTorch LSTM controls while preserving historical config behavior
- recurrent evaluation now has merged compatibility-default block-reset and opt-in streaming semantics
- config/result labeling now records exact WT2 parameter counts and makes tanh explicit for future WT103 Neo runs
- a GPT-2-style Transformer control is merged with aligned Torch/MLX causal checkpoint behavior
- Torch LSTM optimizer grouping now aligns backend-specific recurrent parameter names
- the explicit Torch LSTM single-bias mode now matches MLX trainable parameter
  and effective-bias update semantics
- deterministic MLX/Torch LSTM forward, recurrent-state, loss, and checkpoint
  conversion parity now covers one/two layers and all supported norm modes with
  the merged explicit aligned RMSNorm epsilon contract
- deterministic LSTM gradient, optimizer, 12-step fixed-batch trajectory, and
  same-backend optimizer-resume parity is merged
- explicit matched-dropout and standard-init controls now construct on MLX and
  Torch without changing missing-key MLX initialization
- MLX-reference Torch warmup/cosine timing and deterministic public-loop LSTM
  parity are implemented and pending review
- old result rows stay provenance-bound; do not silently reinterpret them after baseline changes

Current active checkpoint:

`LSTM four-way alignment correction queue`

Current implementation plan:

`docs/IMPLEMENTATION_PLAN.md`

## Checkpoint Chain

1. MLX reference contract
   - Status: done
   - Result wanted: MLX stays frozen as reference; PyTorch CPU proves semantic parity on tiny deterministic tests.
2. MPS no-checkpoint parity ladder
   - Status: done
   - Result wanted: PyTorch CPU vs PyTorch MPS proves forward, backward, optimizer, training-trajectory, checkpoint, and memory behavior without activation checkpointing.
3. MPS endurance classification
   - Status: done
   - Result wanted: MPS memory behavior is classified as flat, bounded, linear growth, or superlinear growth under a safe local probe.
4. CUDA harness preparation
   - Status: done
   - Result wanted: CUDA tests are skip-safe without hardware and reuse the same CPU bridge.
5. External Nvidia CUDA validation
   - Status: blocked outside the local queue
   - Result wanted: CUDA parity is claimed only after the opt-in CUDA probe runs, not skips, on a real Nvidia CUDA machine.
6. Baseline taxonomy and LSTM strengthening
   - Status: done
   - Result wanted: LSTM is named as a normalized/matched recurrent control, and a stronger LSTM variant covers standard init and dropout-policy concerns.
7. Evaluation semantics and config/result labels
   - Status: done
   - Result wanted: recurrent block-reset versus streaming-state eval is explicit, WT2 labels stop implying inaccurate parameter counts, and Neo tanh activation runs are labeled separately from older custom-activation runs.
8. Transformer control strengthening
   - Status: done
   - Result wanted: Transformer comparisons use either a clearly limited internal control or a GPT-2-style baseline with modern enough defaults.
9. LSTM optimizer grouping parity
   - Status: done
   - Result wanted: MLX-reference Torch LSTM training applies recurrent, projection, embedding, and zero-decay buckets by parameter role without changing Neo or Transformer behavior.
10. LSTM four-way parity and aligned baseline profile
   - Status: in progress
   - Result wanted: MLX/Torch LSTM has an explicit same-weight parameter,
     forward, gradient, optimizer, trajectory, and checkpoint contract; the
     matched and standard-init profiles remove accidental LSTM-only dropout
     without changing historical run interpretation.

## Remaining Scale Estimate

- PR #27 is implemented and pending review; 1 later local alignment PR remains queued.
- Paper-quality result production remains blocked until the LSTM four-way
  alignment exit passes and a separate run plan is approved.
- CUDA validation remains blocked on access to Nvidia hardware or a provisioned GPU runner.
- Reproducers must run `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py` and confirm it does not skip before making CUDA claims.
- Standard GitHub-hosted runners for individual repos are not an acceptable substitute for Nvidia GPU validation.
- WT103 revalidation is intentionally outside the active checkpoint chain.

## Update Rules

Update this file only when checkpoint state changes. Routine task detail belongs in `docs/IMPLEMENTATION_PLAN.md`; durable decisions belong in `docs/DEVLOG.md`.
