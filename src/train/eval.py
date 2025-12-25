"""Evaluation helpers."""

import math
from typing import Any, Dict, Optional

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


class ActSparsityMeter:
    def __init__(self, eps: float = 0.0):
        self.eps = eps
        self.zeros = 0
        self.total = 0
        self.handles = []

    def _accum(self, t: torch.Tensor):
        t = t.detach()
        if self.eps <= 0:
            z = (t == 0).sum().item()
        else:
            z = (t.abs() <= self.eps).sum().item()
        self.zeros += z
        self.total += t.numel()

    def attach_recurrent(self, model: torch.nn.Module):
        def hook_recurrent(_, __, out):
            y = out[0] if isinstance(out, tuple) else out
            self._accum(y)

        target = getattr(model, "recurrent", None)
        if target is None:
            target = getattr(model, "lstm", None)
        if target is None:
            raise AttributeError("Model does not expose a recurrent module.")
        self.handles.append(target.register_forward_hook(hook_recurrent))

        def hook_head(_, __, out):
            tgt = out if not isinstance(out, tuple) else out[0]
            self._accum(tgt)

        projection_module = getattr(model, "out_proj", None) or getattr(model, "drop", None)
        if projection_module is not None:
            self.handles.append(projection_module.register_forward_hook(hook_head))

    def attach_transformer(self, model: torch.nn.Module):
        for block in model.blocks:
            def _hook_accum(_, __, output, *, meter=self):
                tgt = output[0] if isinstance(output, tuple) else output
                meter._accum(tgt)

            self.handles.append(block.attn.register_forward_hook(_hook_accum))
            self.handles.append(block.mlp.register_forward_hook(_hook_accum))

        def hook_norm(_, __, out):
            self._accum(out)

        self.handles.append(model.ln_f.register_forward_hook(hook_norm))

    def summary(self) -> float:
        return (self.zeros / self.total) if self.total > 0 else 0.0

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


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


def measure_activation_sparsity(
    model: torch.nn.Module,
    ids: torch.Tensor,
    cfg: Any,
    device: torch.device,
    max_batches: int = 50,
) -> Optional[float]:
    meter = ActSparsityMeter(eps=0.0)
    model.eval()
    T = int(_cfg_get(cfg, "block_size", 128))
    B = min(int(_cfg_get(cfg, "batch_size", 16)), 16)
    with torch.no_grad():
        if hasattr(model, "blocks"):
            meter.attach_transformer(model)
            N = ids.size(0)
            batches = 0
            for start in range(0, N - (T + 1), T * B):
                if batches >= max_batches:
                    break
                cur_B = min(B, (N - (start + T + 1)) // T)
                if cur_B <= 0:
                    break
                x = torch.stack([ids[start + i * T: start + i * T + T] for i in range(cur_B)]).to(device)
                _ = model(x.t().contiguous(), None)
                batches += 1
        else:
            meter.attach_recurrent(model)
            N = ids.size(0)
            batches = 0
            for start in range(0, N - (T + 1), T * B):
                if batches >= max_batches:
                    break
                cur_B = min(B, (N - (start + T + 1)) // T)
                if cur_B <= 0:
                    break
                x = torch.stack([ids[start + i * T: start + i * T + T] for i in range(cur_B)]).to(device)
                state = init_state_for_model(model, cur_B, device)
                _ = model(x.t().contiguous(), state)
                batches += 1
    sparsity = meter.summary()
    meter.clear()
    return sparsity


def profile_real_flops_recurrent(model: torch.nn.Module, cfg: Any) -> Optional[float]:
    try:
        from thop import profile
        import copy

        m = copy.deepcopy(model).to("cpu").eval()
        T = min(int(_cfg_get(cfg, "block_size", 128)), 128)
        B = 8
        idx = torch.randint(0, int(_cfg_get(cfg, "vocab_size", 0)), (T, B), dtype=torch.long)
        state = init_state_for_model(m, B, torch.device("cpu"))
        macs, _ = profile(m, inputs=(idx, state), verbose=False)
        flops = 2 * macs
        return (flops / (B * T)) / 1e9
    except Exception:
        return None


def profile_real_flops_transformer(model: torch.nn.Module, cfg: Any) -> Optional[float]:
    try:
        from thop import profile
        import copy
        import torch.nn as nn

        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, idx):  # type: ignore[override]
                return self.inner(idx, None)[0]

        m = copy.deepcopy(model).to("cpu").eval()
        T = min(int(_cfg_get(cfg, "block_size", 128)), 128)
        T = min(T, m.pos_emb.num_embeddings)
        B = 8
        idx = torch.randint(0, int(_cfg_get(cfg, "vocab_size", 0)), (T, B), dtype=torch.long)
        wrapper = Wrapper(m)
        macs, _ = profile(wrapper, inputs=(idx,), verbose=False)
        flops = 2 * macs
        return (flops / (B * T)) / 1e9
    except Exception:
        return None


def evaluate_metrics(model: torch.nn.Module, ids: torch.Tensor, cfg: Any, device: torch.device) -> Dict[str, Optional[float]]:
    ppl = eval_perplexity(model, ids, cfg, device)
    sparsity = measure_activation_sparsity(model, ids, cfg, device, max_batches=50)
    if hasattr(model, "blocks"):
        gflops = profile_real_flops_transformer(model, cfg)
    else:
        gflops = profile_real_flops_recurrent(model, cfg)
    return {
        "ppl": ppl,
        "act_sparsity": sparsity,
        "gflops_per_token": gflops,
    }
