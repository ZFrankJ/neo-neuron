# Requirements

This repo uses PyTorch plus a small set of supporting libraries for data loading,
training, and analysis. Install the base requirements first, then add optional
packages as needed.

## Core (required)
- torch
- datasets
- transformers
- pyyaml
- numpy

## Training utilities
- psutil (memory reporting)

## Analysis / probing (optional but used by scripts)
- matplotlib (probe plots)
- thop (FLOPs profiling)

## Optional backend
- mlx (Apple Silicon / macOS only; enables `backend: mlx`)

## Install
```
pip install -r requirements.txt
```

For macOS Apple Silicon MLX parity work, install with the bounded parity
constraints:

```
pip install -c requirements-lock/constraints-macos-mlx.txt -r requirements.txt
pip install -c requirements-lock/constraints-macos-mlx.txt pytest
```

The macOS parity constraints keep MLX and MLX-Metal pinned to the tested
reference versions while allowing compatible minor/patch releases for the
general Python stack. Linux CI intentionally does not use this constraints file
so it is not tied to macOS-only MLX wheels.
