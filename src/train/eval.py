"""Evaluation helpers."""

import math
from typing import Any

import torch
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def init_state_for_model(model, batch_size: int, device: torch.device):
    if hasattr(model, "init_state"):
        return model.init_state(batch_size, device)
    return None


def eval_perplexity(model: torch.nn.Module, ids: torch.Tensor, cfg: Any, device: torch.device) -> float:
    model.eval()
    T = int(_cfg_get(cfg, "block_size", 128))
    B = int(_cfg_get(cfg, "batch_size", 16))
    vocab_size = int(_cfg_get(cfg, "vocab_size", 0))
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        N = ids.size(0)
        for start in range(0, N - (T + 1), T * B):
            cur_B = min(B, (N - (start + T + 1)) // T)
            if cur_B <= 0:
                break
            x = torch.stack([ids[start + i * T: start + i * T + T] for i in range(cur_B)]).to(device)
            y = torch.stack([ids[start + i * T + 1: start + i * T + T + 1] for i in range(cur_B)]).to(device)
            x = x.t().contiguous()
            y = y.t().contiguous()
            state = init_state_for_model(model, cur_B, device)
            logits, _ = model(x, state)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += cur_B * T
    return math.exp(total_loss / max(1, total_tokens))
