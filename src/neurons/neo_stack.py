"""Cortical recurrent stack for Neo models."""

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .neo_cell import BaseCorticalNeuron, CorticalNeuronModeC

CELL_REGISTRY: Dict[str, Type[BaseCorticalNeuron]] = {
    "mode_c": CorticalNeuronModeC,
}


class CorticalRecurrentStack(nn.Module):
    def __init__(self, d_model, n_layers, cell_type, cell_kwargs, use_checkpoint=False):
        super().__init__()
        if cell_type not in CELL_REGISTRY:
            raise ValueError(f"Unsupported cell_type '{cell_type}'.")
        cell_cls = CELL_REGISTRY[cell_type]
        self.layers = nn.ModuleList([cell_cls(d_model, d_model, **cell_kwargs) for _ in range(n_layers)])
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

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        T, B, D = x.shape
        y = torch.empty((T, B, D), device=x.device, dtype=x.dtype)

        if state is not None:
            if state.dim() != 3 or state.size(0) != self.n_layers or state.size(2) != self.d_model:
                raise ValueError(
                    f"Expected state of shape [layers({self.n_layers}), batch, d_model({self.d_model})], "
                    f"got {tuple(state.shape)}"
                )
            prev_states = list(state.unbind(0))
        else:
            prev_states = [None] * self.n_layers

        for t in range(T):
            h = x[t]
            for i, layer in enumerate(self.layers):
                ps = prev_states[i]
                reset_flag = state is None and t == 0

                def step(h_in, ps_in):
                    out, ns, _ = layer(h_in, prev_state=ps_in, reset=reset_flag)
                    return out, ns

                use_ckpt = self.training and self.use_checkpoint and h.requires_grad and ps is not None
                if use_ckpt:
                    h, ns = checkpoint(step, h, ps, use_reentrant=False)
                else:
                    h, ns = step(h, ps)

                prev_states[i] = ns

            y[t] = h

        new_state = torch.stack(prev_states, dim=0)
        if state is None:
            for i, layer in enumerate(self.layers):
                layer.prev_state = new_state[i].detach()
        return y, new_state
