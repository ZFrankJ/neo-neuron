"""Cortical recurrent stack for Neo models."""

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .neo_cell import BaseCorticalNeuron, CorticalNeuron

CELL_REGISTRY: Dict[str, Type[BaseCorticalNeuron]] = {
    "cortical": CorticalNeuron,
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
        self.last_fx_energy = None

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
        energy_sum = None
        energy_count = 0

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
                    out, ns, aux = layer(h_in, prev_state=ps_in, reset=reset_flag)
                    fx_energy = None
                    if isinstance(aux, dict):
                        fx_raw = aux.get("f_x_raw")
                        if isinstance(fx_raw, torch.Tensor):
                            fx_energy = fx_raw.pow(2).mean()
                    if fx_energy is None:
                        fx_energy = h_in.new_zeros(())
                    return out, ns, fx_energy

                use_ckpt = self.training and self.use_checkpoint and h.requires_grad and ps is not None
                if use_ckpt:
                    h, ns, fx_energy = checkpoint(step, h, ps, use_reentrant=False)
                else:
                    h, ns, fx_energy = step(h, ps)

                prev_states[i] = ns
                if energy_sum is None:
                    energy_sum = fx_energy
                else:
                    energy_sum = energy_sum + fx_energy
                energy_count += 1

            y[t] = h

        new_state = torch.stack(prev_states, dim=0)
        if energy_sum is None:
            self.last_fx_energy = None
        else:
            self.last_fx_energy = energy_sum / max(1, energy_count)
        if state is None:
            for i, layer in enumerate(self.layers):
                layer.prev_state = new_state[i].detach()
        return y, new_state
