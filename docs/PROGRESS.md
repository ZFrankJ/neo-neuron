# Progress

Checkpoint tracker for the active large goal. This file sits between `docs/ROADMAP.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Active Large Goal

Roadmap priority:

`4. Result Production And WT103 Revalidation`

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
  parity are merged
- a checked-in Wikitext-2 trial profile now freezes the standard-init,
  no-layer-dropout, single-bias, streaming-eval contract with equal Torch/MLX
  trainable parameter counts and a 10%-warmup cosine schedule
- old result rows stay provenance-bound; do not silently reinterpret them after baseline changes
- a four-path code and config audit confirmed that clean MLX Neo and LSTM runs
  remain usable under their historical labels, while plain Torch backend
  overrides of the WT103 configs are not aligned experiments
- WT103 revalidation planning and new-profile config preparation are now
  approved, but no new training process has been authorized or started
- clean MLX Neo checkpoints do not require retraining for fields that were
  already effective on MLX; streaming checkpoint reevaluation remains the
  authoritative path
- Neo MLX now rejects unsupported explicit RMSNorm epsilon values, tanh is in
  the committed mapped-weight parity matrix, and MLX device state is isolated
  across the focused parity suite

Current active checkpoint:

`Neo MLX contract hardening complete; WT103 diagnostic profile preparation pending PR #29 merge`

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
   - Status: done
   - Result wanted: MLX/Torch LSTM has an explicit same-weight parameter,
     forward, gradient, optimizer, trajectory, and checkpoint contract; the
     matched and standard-init profiles remove accidental LSTM-only dropout
     without changing historical run interpretation.
11. Four-path fairness audit and WT103 revalidation decision
   - Status: done
   - Result wanted: distinguish mapped-weight backend parity from fresh-run
     equivalence, preserve clean historical MLX results, and identify the minimum
     new experiment that tests the LSTM depth-regularization confound.
12. Neo MLX contract and parity-test hardening
   - Status: done
   - Result wanted: reject unsupported explicit Neo MLX RMSNorm epsilon values,
     cover tanh in the committed mapped-weight parity matrix, and prevent MLX
     device state from leaking between tests.
13. WT103 diagnostic profile preparation
   - Status: pending PR #29 merge
   - Result wanted: add separately named, test-covered 50M-recurrent-core,
     approximately 60M-total matched and standard-init no-layer-dropout LSTM
     configs without editing historical paths.
14. 50M-recurrent-core matched-no-layer-dropout LSTM diagnostic
   - Status: pending explicit run start after checkpoint 13 merges
   - Result wanted: train with a 12-epoch scheduler contract, inspect at epoch 4,
     and continue, switch profile, or stop according to the predeclared gate.

## Remaining Scale Estimate

- One implementation PR remains before the first diagnostic: new WT103 profile
  configs plus config tests after PR #29 merges.
- The diagnostic run is not a code PR and must not start before both
  implementation PRs merge and the user explicitly starts or authorizes it.
- A full LSTM scaling rebuild remains conditional on meaningful 50M-recurrent-core
  recovery; no Neo retraining is planned.
- CUDA validation remains blocked on access to Nvidia hardware or a provisioned GPU runner.
- Reproducers must run `NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py` and confirm it does not skip before making CUDA claims.
- Standard GitHub-hosted runners for individual repos are not an acceptable substitute for Nvidia GPU validation.
- PyTorch MPS remains outside scientific result production, and historical
  PyTorch MPS rows remain unusable.

## Update Rules

Update this file only when checkpoint state changes. Routine task detail belongs in `docs/IMPLEMENTATION_PLAN.md`; durable decisions belong in `docs/DEVLOG.md`.
