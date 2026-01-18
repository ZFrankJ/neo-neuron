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
        alpha_init: float = 1e-1,
        alpha_trainable: bool = True,
        alpha_min: float = 1e-2,
        alpha_max: float = 1e0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        alpha_value = torch.full((self.output_dim,), float(alpha_init))
        if alpha_trainable:
            self.alpha = nn.Parameter(alpha_value)
        else:
            self.register_buffer("alpha", alpha_value)
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

    def _alpha_value(self, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, min=self.alpha_min, max=self.alpha_max)
        if like is not None:
            alpha = alpha.to(device=like.device, dtype=like.dtype)
        return alpha


class CorticalNeuronModeC(BaseCorticalNeuron):
    def __init__(
        self,
        input_dim,
        output_dim,
        alpha_init: float = 1e-1,
        alpha_trainable: bool = True,
        alpha_min: float = 1e-2,
        alpha_max: float = 1e0,
    ):
        super().__init__(
            input_dim,
            output_dim,
            alpha_init=alpha_init,
            alpha_trainable=alpha_trainable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        self.fg_linear = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x, prev_state=None, reset=False):
        s_prev = self._resolve_prev_state(x, prev_state, reset)
        fg = self.fg_linear(x)
        f_x, g_out = fg.chunk(2, dim=-1)
        output, state = fused_cortical_step(f_x, s_prev, g_out, self._alpha_value(f_x))

        if prev_state is None:
            self.prev_state = state.detach()

        return output, state, f_x
