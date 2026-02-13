"""Cortical recurrent stack for Neo models."""

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .neo_cell import BaseCorticalNeuron, CorticalNeuron

CELL_REGISTRY: Dict[str, Type[BaseCorticalNeuron]] = {
    "cortical": CorticalNeuron,
}


def _build_output_norm(norm_type: str, d_model: int) -> nn.Module:
    norm = str(norm_type).strip().lower()
    if norm in ("none", "off", "identity"):
        return nn.Identity()
    if norm in ("layernorm", "layer_norm", "ln"):
        return nn.LayerNorm(d_model)
    if norm in ("rmsnorm", "rms_norm", "rms"):
        rms_norm = getattr(nn, "RMSNorm", None)
        if rms_norm is None:
            raise ValueError("RMSNorm is not available in this torch version.")
        return rms_norm(d_model)
    raise ValueError(f"Unsupported output_norm '{norm_type}'.")


class CorticalRecurrentStack(nn.Module):
    def __init__(
        self,
        d_model,
        n_layers,
        cell_type,
        cell_kwargs,
        use_checkpoint=False,
        output_norm: str = "layernorm",
    ):
        super().__init__()
        if cell_type not in CELL_REGISTRY:
            raise ValueError(f"Unsupported cell_type '{cell_type}'.")
        cell_kwargs = dict(cell_kwargs or {})
        # Norm is now handled at stack level (pre-layer + final norm), not in the cell.
        cell_kwargs.pop("output_norm", None)
        cell_cls = CELL_REGISTRY[cell_type]
        self.layers = nn.ModuleList([cell_cls(d_model, d_model, **cell_kwargs) for _ in range(n_layers)])
        self.pre_norms = nn.ModuleList([_build_output_norm(output_norm, d_model) for _ in range(n_layers)])
        self.stack_norm = _build_output_norm(output_norm, d_model)
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.use_checkpoint = bool(use_checkpoint)

    @torch.no_grad()
    def reset_state(self, batch_size: int, device=None, dtype=None):
        for layer in self.layers:
            layer.reset_state(batch_size, device=device, dtype=dtype)

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        device = device if device is not None else self.layers[0].fg_linear.weight.device
        dtype = dtype if dtype is not None else self.layers[0].fg_linear.weight.dtype
        self.reset_state(batch_size, device=device, dtype=dtype)
        return torch.zeros(self.n_layers, batch_size, self.d_model, device=device, dtype=dtype)

    def _resolve_prev_states(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
    ):
        if state is not None:
            if state.dim() != 3 or state.size(0) != self.n_layers or state.size(2) != self.d_model:
                raise ValueError(
                    f"Expected state of shape [layers({self.n_layers}), batch, d_model({self.d_model})], "
                    f"got {tuple(state.shape)}"
                )
            return list(state.unbind(0))
        return [None] * self.n_layers

    def _forward_stepwise(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        T, B, D = x.shape
        y = torch.empty((T, B, D), device=x.device, dtype=x.dtype)
        prev_states = self._resolve_prev_states(x, state)

        for t in range(T):
            h = x[t]
            for i, layer in enumerate(self.layers):
                ps = prev_states[i]
                reset_flag = state is None and t == 0
                h_norm = self.pre_norms[i](h)

                def step(h_in, ps_in):
                    out, ns, _ = layer(h_in, prev_state=ps_in, reset=reset_flag)
                    return out, ns

                use_ckpt = self.training and self.use_checkpoint and h_norm.requires_grad and ps is not None
                if use_ckpt:
                    h, ns = checkpoint(step, h_norm, ps, use_reentrant=False)
                else:
                    h, ns = step(h_norm, ps)

                prev_states[i] = ns

            h = self.stack_norm(h)
            y[t] = h

        new_state = torch.stack(prev_states, dim=0)
        if state is None:
            for i, layer in enumerate(self.layers):
                layer.prev_state = new_state[i].detach()
        return y, new_state
