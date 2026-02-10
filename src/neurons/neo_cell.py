"""Cortical neuron cell definitions."""

from typing import Optional

import torch
import torch.nn as nn

from .activations import cortical_activation, fused_cortical_step


class BaseCorticalNeuron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation_id = 3
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
        return cortical_activation(x, activation_id=self.activation_id)


def _build_output_norm(norm_type: str, output_dim: int) -> nn.Module:
    norm = str(norm_type).strip().lower()
    if norm in ("none", "off", "identity"):
        return nn.Identity()
    if norm in ("layernorm", "layer_norm", "ln"):
        return nn.LayerNorm(output_dim)
    if norm in ("rmsnorm", "rms_norm", "rms"):
        rms_norm = getattr(nn, "RMSNorm", None)
        if rms_norm is None:
            raise ValueError("RMSNorm is not available in this torch version.")
        return rms_norm(output_dim)
    raise ValueError(f"Unsupported output_norm '{norm_type}'.")


def _parse_activation_id(activation_id) -> int:
    if isinstance(activation_id, str):
        text = activation_id.strip().lower()
        if text.startswith("id"):
            text = text[2:]
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"Unsupported activation_id '{activation_id}'. Expected id3/id4/id5.") from exc
    else:
        value = int(activation_id)
    if value not in (3, 4, 5):
        raise ValueError(f"Unsupported activation_id '{activation_id}'. Expected id3/id4/id5.")
    return value


class CorticalNeuron(BaseCorticalNeuron):
    def __init__(
        self,
        input_dim,
        output_dim,
        output_norm: str = "layernorm",
        activation_id="id3",
    ):
        super().__init__(input_dim, output_dim)
        self.fg_linear = nn.Linear(input_dim, 2 * output_dim)
        self.output_norm_type = str(output_norm)
        self.out_norm = _build_output_norm(self.output_norm_type, output_dim)
        self.activation_id = _parse_activation_id(activation_id)

    def forward(self, x, prev_state=None, reset=False):
        s_prev = self._resolve_prev_state(x, prev_state, reset)
        fg = self.fg_linear(x)
        f_x_raw, g_x_raw = fg.chunk(2, dim=-1)
        output, state = fused_cortical_step(f_x_raw, s_prev, g_x_raw, self.activation_id)
        output = self.out_norm(output)

        if prev_state is None:
            self.prev_state = state.detach()

        # Keep raw traces only for eval/probe to avoid per-step dict allocations during training.
        aux = None if self.training else {"f_x_raw": f_x_raw, "g_x_raw": g_x_raw}
        return output, state, aux
