"""Batching helpers for autoregressive LMs."""

import torch


_OFFSET_CACHE = {}


def _get_offsets(block_size: int, device: torch.device) -> torch.Tensor:
    key = (block_size, device)
    offsets = _OFFSET_CACHE.get(key)
    if offsets is None or offsets.device != device:
        offsets = torch.arange(block_size + 1, device=device)
        _OFFSET_CACHE[key] = offsets
    return offsets


def get_batch(ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    n_tokens = ids.size(0)
    max_start = max(1, n_tokens - block_size - 1)
    starts = torch.randint(low=0, high=max_start, size=(batch_size,), device=ids.device)
    offsets = _get_offsets(block_size, device=ids.device)
    idx = starts[:, None] + offsets[None, :]
    seq = ids[idx]  # [B, T+1]
    x = seq[:, :-1].t().contiguous()
    y = seq[:, 1:].t().contiguous()
    if device != ids.device:
        x = x.to(device)
        y = y.to(device)
    return x, y  # [T, B]
