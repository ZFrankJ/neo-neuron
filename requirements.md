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

## Install
```
pip install -r requirements.txt
```
