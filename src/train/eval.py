"""Evaluation helpers."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from src.runtime.eval_semantics import resolve_eval_regime


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def init_state_for_model(model, batch_size: int, device: torch.device):
    if hasattr(model, "init_state"):
        return model.init_state(batch_size, device)
    return None


def _eval_batches(ids: torch.Tensor, block_size: int, batch_size: int, regime: str):
    if regime == "streaming":
        batch_size = min(batch_size, max(1, (ids.size(0) - 1) // block_size))
        lane_len = ids.size(0) // batch_size
        usable_steps = ((lane_len - 1) // block_size) * block_size
        if usable_steps <= 0:
            return
        lanes = ids[: lane_len * batch_size].reshape(batch_size, lane_len)
        for start in range(0, usable_steps, block_size):
            yield (
                lanes[:, start : start + block_size].t().contiguous(),
                lanes[:, start + 1 : start + block_size + 1].t().contiguous(),
                batch_size,
            )
        return

    n_tokens = ids.size(0)
    for start in range(0, n_tokens - (block_size + 1), block_size * batch_size):
        cur_batch = min(batch_size, (n_tokens - (start + block_size + 1)) // block_size)
        if cur_batch <= 0:
            break
        x = torch.stack([ids[start + i * block_size : start + i * block_size + block_size] for i in range(cur_batch)])
        y = torch.stack([ids[start + i * block_size + 1 : start + i * block_size + block_size + 1] for i in range(cur_batch)])
        yield x.t().contiguous(), y.t().contiguous(), cur_batch


def activation_sparsity_eps(cfg: Any) -> float:
    return float(_cfg_get(cfg, "activation_sparsity_eps", 1e-2))


class ActSparsityMeter:
    def __init__(self, eps: float = 0.0):
        self.eps = eps
        self.step_sparsity_sum = 0.0
        self.step_count = 0
        self.handles = []

    def _accum(self, t: torch.Tensor):
        t = t.detach()
        if t.numel() == 0:
            return
        # Measure sparsity per time step (or per emitted slice) and average,
        # rather than pooling exact-zero counts over every element globally.
        if t.dim() >= 2:
            flat = t.reshape(t.shape[0], -1)
        else:
            flat = t.reshape(1, -1)
        if self.eps <= 0:
            frac = (flat == 0).float().mean(dim=1)
        else:
            frac = (flat.abs() <= self.eps).float().mean(dim=1)
        self.step_sparsity_sum += frac.sum().item()
        self.step_count += int(frac.numel())

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
        return (self.step_sparsity_sum / self.step_count) if self.step_count > 0 else 0.0

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def eval_perplexity(model: torch.nn.Module, ids: torch.Tensor, cfg: Any, device: torch.device) -> float:
    model.eval()
    T = int(_cfg_get(cfg, "block_size", 128))
    B = int(_cfg_get(cfg, "batch_size", 16))
    vocab_size = int(_cfg_get(cfg, "vocab_size", 0))
    regime = resolve_eval_regime(cfg)
    total_loss = 0.0
    total_tokens = 0
    state = None
    with torch.no_grad():
        for x, y, cur_B in _eval_batches(ids, T, B, regime):
            x = x.to(device)
            y = y.to(device)
            if regime == "block_reset" or state is None:
                state = init_state_for_model(model, cur_B, device)
            logits, state_out = model(x, state)
            state = state_out if regime == "streaming" else None
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
    meter = ActSparsityMeter(eps=activation_sparsity_eps(cfg))
    model.eval()
    T = int(_cfg_get(cfg, "block_size", 128))
    B = min(int(_cfg_get(cfg, "batch_size", 16)), 16)
    regime = resolve_eval_regime(cfg)
    with torch.no_grad():
        if hasattr(model, "blocks"):
            meter.attach_transformer(model)
            batches = 0
            for x, _, _ in _eval_batches(ids, T, B, regime):
                if batches >= max_batches:
                    break
                _ = model(x.to(device), None)
                batches += 1
        else:
            meter.attach_recurrent(model)
            batches = 0
            state = None
            for x, _, cur_B in _eval_batches(ids, T, B, regime):
                if batches >= max_batches:
                    break
                if regime == "block_reset" or state is None:
                    state = init_state_for_model(model, cur_B, device)
                _, state_out = model(x.to(device), state)
                state = state_out if regime == "streaming" else None
                batches += 1
    sparsity = meter.summary()
    meter.clear()
    return sparsity


def profile_real_flops_recurrent(model: torch.nn.Module, cfg: Any) -> Optional[float]:
    try:
        flops_per_token = 0.0

        def _linear_weight_flops(module: Any) -> float:
            weight = getattr(module, "weight", None)
            if weight is None:
                return 0.0
            return float(2 * weight.numel())

        vocab_size = int(_cfg_get(cfg, "vocab_size", 0))

        in_proj = getattr(model, "in_proj", None)
        if in_proj is not None:
            flops_per_token += _linear_weight_flops(in_proj)

        out_proj = getattr(model, "out_proj", None)
        if out_proj is not None:
            flops_per_token += _linear_weight_flops(out_proj)

        head = getattr(model, "head", None)
        if head is not None:
            flops_per_token += _linear_weight_flops(head)
        else:
            emb = getattr(model, "emb", None)
            if emb is not None and hasattr(emb, "weight"):
                flops_per_token += float(2 * emb.weight.numel())

        lstm = getattr(model, "lstm", None)
        if lstm is not None:
            for li in range(int(getattr(lstm, "num_layers", 0))):
                w_ih = getattr(lstm, f"weight_ih_l{li}", None)
                w_hh = getattr(lstm, f"weight_hh_l{li}", None)
                if w_ih is not None:
                    flops_per_token += float(2 * w_ih.numel())
                if w_hh is not None:
                    flops_per_token += float(2 * w_hh.numel())
            return flops_per_token / 1e9

        recurrent = getattr(model, "recurrent", None)
        if recurrent is not None and hasattr(recurrent, "layers"):
            for layer in recurrent.layers:
                fg_linear = getattr(layer, "fg_linear", None)
                if fg_linear is not None:
                    flops_per_token += _linear_weight_flops(fg_linear)
            return flops_per_token / 1e9

        return None
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


def evaluate_metrics(model: torch.nn.Module, ids: torch.Tensor, cfg: Any, device: torch.device) -> Dict[str, Any]:
    regime = resolve_eval_regime(cfg)
    ppl = eval_perplexity(model, ids, cfg, device)
    sparsity = measure_activation_sparsity(model, ids, cfg, device, max_batches=50)
    if hasattr(model, "blocks"):
        gflops = profile_real_flops_transformer(model, cfg)
    else:
        gflops = profile_real_flops_recurrent(model, cfg)
    return {
        "eval_regime": regime,
        "ppl": ppl,
        "act_sparsity": sparsity,
        "act_sparsity_eps": activation_sparsity_eps(cfg),
        "gflops_per_token": gflops,
    }
