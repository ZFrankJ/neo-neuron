"""Cortical neuron cell definitions."""

from typing import Optional

import torch
import torch.nn as nn

from .activations import cortical_piecewise_activation


class BaseCorticalNeuron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        exp_factor: float,
        neg_quad: float,
        exp_clip: float,
        eps: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.exp_factor = float(exp_factor)
        self.neg_quad = float(neg_quad)
        self.exp_clip = float(exp_clip)
        self.eps = float(eps)
        self.prev_state: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset_state(self, batch_size: int, device=None, dtype=None) -> None:
        self.prev_state = torch.zeros(
            batch_size,
            self.output_dim,
            device=device,
            dtype=dtype,
        )

    def _resolve_prev_state(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor],
        reset: bool,
    ) -> torch.Tensor:
        if prev_state is not None:
            return prev_state
        B = x.size(0)
        if (
            reset
            or (self.prev_state is None)
            or (self.prev_state.size(0) != B)
            or (self.prev_state.device != x.device)
            or (self.prev_state.dtype != x.dtype)
        ):
            self.reset_state(B, device=x.device, dtype=x.dtype)
        return self.prev_state  # type: ignore[return-value]

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return cortical_piecewise_activation(x, exp_clip=self.exp_clip)


class CorticalNeuronModeC(BaseCorticalNeuron):
    def __init__(self, input_dim, output_dim, exp_factor, neg_quad, exp_clip, eps):
        super().__init__(input_dim, output_dim, exp_factor, neg_quad, exp_clip, eps)
        self.fg_linear = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x, prev_state=None, reset=False):
        s_prev = self._resolve_prev_state(x, prev_state, reset)
        fg = self.fg_linear(x)
        f_x, g_out = fg.chunk(2, dim=-1)
        hidden_state = f_x + s_prev
        state = cortical_piecewise_activation(hidden_state, exp_clip=self.exp_clip)
        output = state * g_out

        if prev_state is None:
            self.prev_state = state.detach()

        return output, state, f_x
