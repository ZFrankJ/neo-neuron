# Roadmap

This roadmap lists broad project priorities only. Checkpoint state belongs in `docs/PROGRESS.md`; strict task and PR execution belongs in `docs/IMPLEMENTATION_PLAN.md`.

## Priority Order

## 1. Backend Parity Against MLX

- Status: active
- Why now:
  - MLX has produced the trusted result line, while older PyTorch MPS runs showed numerical and memory contamination risk.
  - PyTorch is still needed as the bridge toward CUDA, but it must prove parity against MLX in controlled steps first.
- Focus:
  - Preserve MLX as the frozen reference.
  - Keep PyTorch CPU as the semantic bridge.
  - Build PyTorch MPS no-checkpoint parity before transferring the same harness to CUDA.
- Exit condition:
  - The MPS parity exit in `docs/IMPLEMENTATION_PLAN.md` is satisfied, and CUDA preparation is skip-safe on non-CUDA machines.
- Expanded plan:
  - `docs/PROGRESS.md`
  - `docs/IMPLEMENTATION_PLAN.md`

## 2. CUDA Parity And Runner Enablement

- Status: next
- Why now:
  - The long-term target is PyTorch CUDA parity against MLX, but this repo does not currently have local CUDA hardware.
- Focus:
  - Reuse the PyTorch CPU bridge and parity audit harness.
  - Keep CUDA tests full precision, no checkpoint, no compile, and no fused optimizer until baseline parity is proven.
  - Keep Nvidia GitHub CI optional until a GPU runner is intentionally provisioned.
- Exit condition:
  - CUDA parity tests run on an Nvidia device and pass the same small-scale parity envelope used for MPS.
- Expanded plan:
  - `docs/IMPLEMENTATION_PLAN.md`

## 3. Result Production And WT103 Revalidation

- Status: later
- Why now:
  - Old PyTorch MPS WT103 rows are not trusted, but revalidating large training is expensive and should wait for backend parity evidence.
- Focus:
  - Do not rerun WT103 or rehabilitate old PyTorch result rows until the parity ladder is complete.
- Exit condition:
  - A separate approved plan defines hardware, configs, result labels, and acceptance criteria.
- Expanded plan:
  - not active
