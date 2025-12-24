"""Activation functions for cortical neurons."""

import torch


def cortical_piecewise_activation(x: torch.Tensor, exp_clip: float) -> torch.Tensor:
    x2 = x * x
    pos_part = torch.tanh(x)
    exp_arg = torch.clamp(x, min=-exp_clip, max=0.0)
    neg_poly = (x * x2) / 6.0 - x2 + x
    neg_part = neg_poly * torch.exp(exp_arg)
    return torch.where(x >= 0, pos_part, neg_part)
