"""Activation functions for cortical neurons."""

import torch


def three_state_activation(x: torch.Tensor) -> torch.Tensor:
    x2 = x * x
    pos_part = torch.tanh(x)
    exp_arg = torch.clamp(x, max=0.0)
    neg_poly = (x * x2) / 6.0 - x2 + x
    neg_part = neg_poly * torch.exp(exp_arg)
    return torch.where(x >= 0, pos_part, neg_part)


def _fused_cortical_step(
    f_x: torch.Tensor,
    s_prev: torch.Tensor,
    g_out: torch.Tensor,
    alpha: torch.Tensor,
):
    hidden = (f_x / alpha) + s_prev
    x2 = hidden * hidden
    pos_part = torch.tanh(hidden)
    exp_arg = torch.clamp(hidden, max=0.0)
    neg_poly = (hidden * x2) / 6.0 - x2 + hidden
    neg_part = neg_poly * torch.exp(exp_arg)
    state = torch.where(hidden >= 0, pos_part, neg_part)
    output = state * g_out
    return output, state


if torch.backends.mps.is_available():
    # TorchScript on MPS can fail at runtime; keep eager for stability.
    fused_cortical_step = _fused_cortical_step
else:
    try:  # TorchScript can fuse some elementwise ops in a single graph.
        fused_cortical_step = torch.jit.script(_fused_cortical_step)
    except Exception:  # pragma: no cover - fallback when TorchScript is unavailable
        fused_cortical_step = _fused_cortical_step
