"""Activation functions for cortical neurons."""

import torch


def _negative_branch(hidden: torch.Tensor, activation_id: int) -> torch.Tensor:
    x2 = hidden * hidden
    x3 = hidden * x2
    if activation_id == 4:
        x4 = x2 * x2
        return (x4 / 6.0) + (x3 / 6.0) - x2 + hidden
    if activation_id == 5:
        x4 = x2 * x2
        x5 = hidden * x4
        return (x5 / 120.0) + (x4 / 6.0) + (x3 / 6.0) - x2 + hidden
    return (x3 / 6.0) - x2 + hidden


def cortical_activation(x: torch.Tensor, activation_id: int = 3) -> torch.Tensor:
    pos_part = torch.tanh(x)
    exp_arg = torch.clamp(x, max=0.0)
    neg_poly = _negative_branch(x, int(activation_id))
    neg_part = neg_poly * torch.exp(exp_arg)
    return torch.where(x >= 0, pos_part, neg_part)


def three_state_activation(x: torch.Tensor) -> torch.Tensor:
    return cortical_activation(x, activation_id=3)


def _fused_cortical_step(
    f_x: torch.Tensor,
    s_prev: torch.Tensor,
    g_out: torch.Tensor,
    activation_id: int,
):
    hidden = f_x + s_prev
    pos_part = torch.tanh(hidden)
    exp_arg = torch.clamp(hidden, max=0.0)
    neg_poly = _negative_branch(hidden, int(activation_id))
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
