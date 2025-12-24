"""Batching helpers for autoregressive LMs."""

import torch


def get_batch(ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    n_tokens = ids.size(0)
    starts = torch.randint(low=0, high=max(1, n_tokens - block_size - 1), size=(batch_size,))
    x = torch.stack([ids[s:s + block_size] for s in starts]).to(device)
    y = torch.stack([ids[s + 1:s + block_size + 1] for s in starts]).to(device)
    return x.t().contiguous(), y.t().contiguous()  # [T, B]
