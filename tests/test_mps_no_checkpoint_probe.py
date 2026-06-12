"""Opt-in PyTorch MPS diagnostics for the no-checkpoint Neo path.

These tests are intentionally excluded from normal CI. They are a tiny local
probe only, not evidence for large WT103 training or checkpointed MPS runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.runtime.backends import torch_backend
from src.train.optim import build_optimizer


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("NEO_RUN_MPS_PROBE") != "1",
        reason="set NEO_RUN_MPS_PROBE=1 to run optional MPS diagnostics",
    ),
    pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="PyTorch MPS backend is not available",
    ),
]


TINY_NO_CHECKPOINT_CFG = {
    "model_name": "neo",
    "vocab_size": 64,
    "d_model": 16,
    "d_embed": 8,
    "n_layers": 2,
    "dropout": 0.0,
    "tie_embeddings": True,
    "cell_type": "cortical",
    "activation_id": "id5",
    "recurrent_norm": "rmsnorm",
    "recurrent_norm_place": "all",
    "rmsnorm_eps": 1e-5,
    "use_checkpoint": False,
    "train_regime": "streaming",
    "stream_state": True,
    "block_size": 8,
    "batch_size": 2,
    "lr": 1e-3,
    "betas": (0.9, 0.95),
    "adam_eps": 1e-8,
    "weight_decay_policy": "table",
    "embed_weight_decay": 0.0,
    "proj_weight_decay": 0.0,
    "recurrent_weight_decay": 0.0,
    "seed": 42,
    "reference_backend": "mlx",
}


def _mps_device() -> torch.device:
    return torch.device("mps")


def _sync_mps() -> None:
    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _mps_allocated_bytes() -> int:
    current_allocated = getattr(torch.mps, "current_allocated_memory", None)
    if current_allocated is None:
        pytest.skip("torch.mps.current_allocated_memory is unavailable")
    return int(current_allocated())


def _deterministic_state_dict(template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(1234)
    state = {}
    for name, tensor in template.items():
        values = rng.normal(0.0, 0.03, size=tuple(tensor.shape)).astype(np.float32)
        if name.endswith("output_bias") or name.endswith(".bias"):
            values.fill(0.0)
        if "norm" in name and name.endswith(".weight"):
            values = 1.0 + rng.normal(0.0, 0.01, size=tuple(tensor.shape)).astype(np.float32)
        state[name] = torch.from_numpy(values)
    return state


def _build_model(device: torch.device) -> torch.nn.Module:
    model = torch_backend.build_model(TINY_NO_CHECKPOINT_CFG, "neo").to(device)
    assert model.recurrent.use_checkpoint is False
    return model


def _tokens(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = TINY_NO_CHECKPOINT_CFG
    t = int(cfg["block_size"])
    b = int(cfg["batch_size"])
    vocab = int(cfg["vocab_size"])
    x_np = ((np.arange(t * b, dtype=np.int64).reshape(t, b) * 7 + 3) % vocab).astype(np.int64)
    y_np = ((x_np + 5) % vocab).astype(np.int64)
    state_np = np.random.default_rng(5678).normal(
        0.0,
        0.02,
        size=(int(cfg["n_layers"]), b, int(cfg["d_model"])),
    ).astype(np.float32)
    return (
        torch.from_numpy(x_np).to(device),
        torch.from_numpy(y_np).to(device),
        torch.from_numpy(state_np).to(device),
    )


def _loss_and_grads(model: torch.nn.Module, device: torch.device) -> tuple[float, dict[str, np.ndarray]]:
    model.train()
    x, y, state = _tokens(device)
    model.zero_grad(set_to_none=True)
    logits, _ = model(x, state)
    loss = F.cross_entropy(logits.reshape(-1, int(TINY_NO_CHECKPOINT_CFG["vocab_size"])), y.reshape(-1))
    loss.backward()
    if device.type == "mps":
        _sync_mps()
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        grads[name] = param.grad.detach().cpu().numpy().copy()
    return float(loss.detach().cpu().item()), grads


def test_optional_no_checkpoint_mps_gradients_match_cpu_tiny_probe():
    cpu_model = _build_model(torch.device("cpu"))
    mps_model = _build_model(_mps_device())
    state = _deterministic_state_dict(cpu_model.state_dict())
    cpu_model.load_state_dict(state)
    mps_model.load_state_dict({name: value.to(_mps_device()) for name, value in state.items()})

    cpu_loss, cpu_grads = _loss_and_grads(cpu_model, torch.device("cpu"))
    mps_loss, mps_grads = _loss_and_grads(mps_model, _mps_device())

    assert abs(cpu_loss - mps_loss) <= 5e-3
    total_diff_sq = 0.0
    total_ref_sq = 0.0
    max_diff = 0.0
    assert set(mps_grads) == set(cpu_grads)
    for name, cpu_grad in cpu_grads.items():
        mps_grad = mps_grads[name]
        assert np.all(np.isfinite(mps_grad)), name
        diff = mps_grad - cpu_grad
        total_diff_sq += float(np.sum(diff * diff))
        total_ref_sq += float(np.sum(cpu_grad * cpu_grad))
        max_diff = max(max_diff, float(np.max(np.abs(diff))))
    rel_l2 = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))
    assert rel_l2 <= 5e-2
    assert max_diff <= 2e-2


def test_optional_no_checkpoint_mps_memory_trend_is_bounded_for_tiny_probe():
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    _sync_mps()

    device = _mps_device()
    model = _build_model(device)
    optimizer = build_optimizer(model, TINY_NO_CHECKPOINT_CFG)
    state = _tokens(device)[2]
    allocated = []

    for step in range(8):
        x, y, _ = _tokens(device)
        x = (x + step * 3) % int(TINY_NO_CHECKPOINT_CFG["vocab_size"])
        y = (y + step * 3) % int(TINY_NO_CHECKPOINT_CFG["vocab_size"])
        optimizer.zero_grad(set_to_none=True)
        logits, next_state = model(x, state)
        loss = F.cross_entropy(logits.reshape(-1, int(TINY_NO_CHECKPOINT_CFG["vocab_size"])), y.reshape(-1))
        loss.backward()
        optimizer.step()
        state = next_state.detach()
        del logits, next_state, loss
        _sync_mps()
        allocated.append(_mps_allocated_bytes())

    assert len(allocated) == 8
    assert all(value >= 0 for value in allocated)
    warm_tail = allocated[3:]
    tail_growth = max(warm_tail) - warm_tail[0]
    assert tail_growth <= 64 * 1024 * 1024
