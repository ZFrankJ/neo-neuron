# Progress

Checkpoint tracker for the active large goal. This file sits between `docs/ROADMAP.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Active Large Goal

Roadmap priority:

`1. Backend Parity Against MLX`

Current position:

- checkpoint 3 of 5 is active
- MLX reference parity, optimizer parity, public training-loop parity, checkpoint metadata guards, CI, a seed optional MPS probe, the backend parity audit harness, and MPS short trajectory parity have merged through PR #12
- remaining work is the MPS parity ladder, then CUDA preparation

Current active checkpoint:

`MPS endurance classification`

Current strict task queue:

`docs/IMPLEMENTATION_PLAN.md`

## Checkpoint Chain

1. MLX reference contract
   - Status: done
   - Result wanted: MLX stays frozen as reference; PyTorch CPU proves semantic parity on tiny deterministic tests.
2. MPS no-checkpoint parity ladder
   - Status: active
   - Result wanted: PyTorch CPU vs PyTorch MPS proves forward, backward, optimizer, training-trajectory, checkpoint, and memory behavior without activation checkpointing.
3. MPS endurance classification
   - Status: pending
   - Result wanted: MPS memory behavior is classified as flat, bounded, linear growth, or superlinear growth under a safe local probe.
4. CUDA harness preparation
   - Status: pending
   - Result wanted: CUDA tests are skip-safe without hardware and reuse the same CPU bridge.
5. Optional Nvidia CI enablement
   - Status: pending
   - Result wanted: GPU CI is manual/optional until a real Nvidia runner exists.

## Remaining Scale Estimate

- Approximately 3 to 5 focused PRs before a meaningful MPS parity decision.
- CUDA validation remains blocked on access to Nvidia hardware or a provisioned GPU runner.
- WT103 revalidation is intentionally outside the active checkpoint chain.

## Update Rules

Update this file only when checkpoint state changes. Routine task detail belongs in `docs/IMPLEMENTATION_PLAN.md`; durable decisions belong in `docs/DEVLOG.md`.
