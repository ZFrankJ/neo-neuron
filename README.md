> **Warning**:
> 1. The codes are only verified on MPS system and check compatibility first if you are using CUDA.
> 2. This repo's code is mainly generated through vibe coding (>=90%), so you should better recheck.
> 3. This is my first time to create GitHub repo for scientific researches, may have mistakes.
> - *If any of the above causes problems and errors in reproducibility to you, I will be very sorry for wasting your time and please let me know, I will fix as soon as I can.*

# Neo Neuron: a new parameter-efficient recurrent cell

This repo is for demonstrating a new recurrent neuron founded by me. 

## This repo is currently still in progress, please wait for training completed and results handled.

## Backend selection

This repo now supports two execution backends behind the same script interface:

- `torch` (default, full feature path)
- `mlx` (Apple Silicon / MLX path for training and eval)

You can choose backend in either config or CLI:

```yaml
backend: torch
```

```bash
python3 scripts/train.py --config configs/wt103/neo_20m.yaml --backend mlx
python3 scripts/eval.py --config configs/wt103/neo_20m.yaml --checkpoint checkpoints/best_wt103_neo_20m.pt --backend mlx
```

Note: probing is currently implemented for `torch` backend.
