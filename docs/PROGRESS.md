# Progress

Checkpoint tracker for the active large goal. This file sits between `docs/ROADMAP.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Active Large Goal

Roadmap priority:

`1. Backend Parity Against MLX`

Current position:

- checkpoint 4 of 5 is active
- MLX reference parity, optimizer parity, public training-loop parity, checkpoint metadata guards, CI, a seed optional MPS probe, the backend parity audit harness, MPS short trajectory parity, and MPS memory slope classification have merged through PR #13
- remaining work is CUDA preparation and optional Nvidia CI enablement

Current active checkpoint:

`CUDA harness preparation`

Current strict task queue:

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
   - Status: active
   - Result wanted: CUDA tests are skip-safe without hardware and reuse the same CPU bridge.
5. Optional Nvidia CI enablement
   - Status: pending
   - Result wanted: GPU CI is manual/optional until a real Nvidia runner exists.

## Remaining Scale Estimate

- Approximately 2 to 4 focused PRs before a meaningful MPS/CUDA parity decision.
- CUDA validation remains blocked on access to Nvidia hardware or a provisioned GPU runner.
- WT103 revalidation is intentionally outside the active checkpoint chain.

## Update Rules

Update this file only when checkpoint state changes. Routine task detail belongs in `docs/IMPLEMENTATION_PLAN.md`; durable decisions belong in `docs/DEVLOG.md`.
