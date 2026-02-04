"""Cortical neuron cell definitions."""

from typing import Optional

import torch
import torch.nn as nn

from .activations import fused_cortical_step, three_state_activation


class BaseCorticalNeuron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
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
        return three_state_activation(x)


class CorticalNeuron(BaseCorticalNeuron):
    def __init__(
        self,
        input_dim,
        output_dim,
        g_clamp_L: float = 1.0,
    ):
        super().__init__(input_dim, output_dim)
        self.fg_linear = nn.Linear(input_dim, 2 * output_dim)
        self.g_clamp_L = float(g_clamp_L)

    def forward(self, x, prev_state=None, reset=False):
        s_prev = self._resolve_prev_state(x, prev_state, reset)
        fg = self.fg_linear(x)
        f_x_raw, g_out = fg.chunk(2, dim=-1)
        f_x = f_x_raw
        output, state = fused_cortical_step(f_x, s_prev, g_out, self.g_clamp_L)

        if prev_state is None:
            self.prev_state = state.detach()

        return output, state, {"f_x_raw": f_x_raw, "g_x_raw": g_out}
