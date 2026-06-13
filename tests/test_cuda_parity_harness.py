"""Skip-safe CUDA parity harness preparation.

The optional CUDA probe is a tiny local baseline only. It is not evidence for
WT103 training, checkpointed execution, torch.compile, fused optimizers, AMP, or
TF32 speed paths.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.runtime.backends import torch_backend
from src.runtime.parity_audit import (
    cuda_device_report,
    device_availability,
    make_backend_parity_report,
)
from src.train.optim import build_optimizer


requires_cuda_probe = pytest.mark.skipif(
    os.environ.get("NEO_RUN_CUDA_PROBE") != "1",
    reason="set NEO_RUN_CUDA_PROBE=1 to run optional CUDA diagnostics",
)

_CUDA_AVAILABILITY = device_availability("cuda")
requires_cuda_backend = pytest.mark.skipif(
    not _CUDA_AVAILABILITY.available,
    reason=_CUDA_AVAILABILITY.skip_reason or "PyTorch CUDA backend is not available",
)


CUDA_BASELINE_CFG = {
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
    "use_compile": False,
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

CUDA_SINGLE_STEP_TOLERANCES = {
    "loss_diff": 5e-4,
    "logits_max_diff": 5e-4,
    "state_max_diff": 5e-4,
    "gradient_max_diff": 1e-3,
    "update_max_diff": 1e-3,
}


def test_cuda_device_report_is_skip_safe_and_baseline_only():
    report = cuda_device_report()
    serialized = report.to_dict()

    assert serialized["availability"]["device"] == "cuda"
    assert isinstance(serialized["availability"]["available"], bool)
    assert serialized["use_checkpoint"] is False
    assert serialized["torch_compile"] is False
    assert serialized["fused_optimizer"] is False
    if not serialized["availability"]["available"]:
        assert serialized["availability"]["skip_reason"]


def test_cuda_baseline_cfg_disables_speed_features():
    assert CUDA_BASELINE_CFG["use_checkpoint"] is False
    assert CUDA_BASELINE_CFG["use_compile"] is False
    assert CUDA_BASELINE_CFG["reference_backend"] == "mlx"


def _trusted_cuda_cfg(**overrides) -> dict:
    cfg = dict(CUDA_BASELINE_CFG)
    cfg.update(overrides)
    if bool(cfg.get("use_checkpoint", False)):
        raise ValueError("trusted CUDA parity probes require use_checkpoint=false")
    if bool(cfg.get("use_compile", False)):
        raise ValueError("trusted CUDA parity probes require use_compile=false")
    return cfg


def _audit_model_shape(cfg: dict) -> dict[str, int | str | bool | float]:
    return {
        "model_name": str(cfg["model_name"]),
        "vocab_size": int(cfg["vocab_size"]),
        "d_model": int(cfg["d_model"]),
        "d_embed": int(cfg["d_embed"]),
        "n_layers": int(cfg["n_layers"]),
        "block_size": int(cfg["block_size"]),
        "batch_size": int(cfg["batch_size"]),
        "activation_id": str(cfg["activation_id"]),
        "recurrent_norm": str(cfg["recurrent_norm"]),
        "rmsnorm_eps": float(cfg["rmsnorm_eps"]),
    }


def _build_model(device: torch.device, cfg: dict | None = None) -> torch.nn.Module:
    cfg = _trusted_cuda_cfg() if cfg is None else _trusted_cuda_cfg(**cfg)
    model = torch_backend.build_model(cfg, "neo").to(device=device, dtype=torch.float32)
    assert model.recurrent.use_checkpoint is False
    return model


def _deterministic_state_dict(
    template: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(1234)
    state = {}
    for name, tensor in template.items():
        values = rng.normal(0.0, 0.03, size=tuple(tensor.shape)).astype(np.float32)
        if name.endswith("output_bias") or name.endswith(".bias"):
            values.fill(0.0)
        if "norm" in name and name.endswith(".weight"):
            values = 1.0 + rng.normal(0.0, 0.01, size=tuple(tensor.shape)).astype(
                np.float32
            )
        state[name] = torch.from_numpy(values)
    return state


def _tokens(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = CUDA_BASELINE_CFG
    t = int(cfg["block_size"])
    b = int(cfg["batch_size"])
    vocab = int(cfg["vocab_size"])
    x_np = ((np.arange(t * b, dtype=np.int64).reshape(t, b) * 7 + 3) % vocab).astype(
        np.int64
    )
    y_np = ((x_np + 5) % vocab).astype(np.int64)
    state_np = (
        np.random.default_rng(5678)
        .normal(
            0.0,
            0.02,
            size=(int(cfg["n_layers"]), b, int(cfg["d_model"])),
        )
        .astype(np.float32)
    )
    return (
        torch.from_numpy(x_np).to(device),
        torch.from_numpy(y_np).to(device),
        torch.from_numpy(state_np).to(device),
    )


def _loss_grads_and_outputs(
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.train()
    x, y, state = _tokens(device)
    model.zero_grad(set_to_none=True)
    logits, next_state = model(x, state)
    loss = F.cross_entropy(
        logits.reshape(-1, int(CUDA_BASELINE_CFG["vocab_size"])), y.reshape(-1)
    )
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        grads[name] = param.grad.detach().cpu().numpy().copy()
    return (
        float(loss.detach().cpu().item()),
        logits.detach().cpu().numpy().copy(),
        next_state.detach().cpu().numpy().copy(),
        grads,
    )


def _named_parameter_arrays(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: param.detach().cpu().numpy().copy()
        for name, param in model.named_parameters()
    }


def test_trusted_cuda_probe_rejects_checkpoint_compile_paths():
    with pytest.raises(ValueError, match="use_checkpoint=false"):
        _trusted_cuda_cfg(use_checkpoint=True)
    with pytest.raises(ValueError, match="use_compile=false"):
        _trusted_cuda_cfg(use_compile=True)


@requires_cuda_probe
@requires_cuda_backend
def test_optional_cuda_no_checkpoint_single_step_parity_baseline():
    cfg = _trusted_cuda_cfg()
    device_report = cuda_device_report(apply_full_precision_baseline=True)
    assert device_report.availability.available is True
    assert device_report.use_checkpoint is False
    assert device_report.torch_compile is False
    assert device_report.fused_optimizer is False
    assert device_report.allow_tf32_matmul is False
    if device_report.allow_tf32_cudnn is not None:
        assert device_report.allow_tf32_cudnn is False

    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda")
    cpu_model = _build_model(cpu_device, cfg)
    cuda_model = _build_model(cuda_device, cfg)
    state = _deterministic_state_dict(cpu_model.state_dict())
    cpu_model.load_state_dict(state)
    cuda_model.load_state_dict(
        {name: value.to(cuda_device) for name, value in state.items()}
    )

    cpu_optimizer = build_optimizer(cpu_model, cfg)
    cuda_optimizer = build_optimizer(cuda_model, cfg)

    cpu_loss, cpu_logits, cpu_state, cpu_grads = _loss_grads_and_outputs(
        cpu_model, cpu_device
    )
    cuda_loss, cuda_logits, cuda_state, cuda_grads = _loss_grads_and_outputs(
        cuda_model, cuda_device
    )

    cpu_optimizer.step()
    cuda_optimizer.step()
    torch.cuda.synchronize(cuda_device)

    report = make_backend_parity_report(
        backend_pair=("torch", "torch"),
        device_pair=("cpu", "cuda"),
        seed=int(cfg["seed"]),
        model_shape=_audit_model_shape(cfg),
        use_checkpoint=bool(cfg["use_checkpoint"]),
        loss_pair=(cuda_loss, cpu_loss),
        logits_pair=(cuda_logits, cpu_logits),
        state_pair=(cuda_state, cpu_state),
        gradient_pair=(cuda_grads, cpu_grads),
        update_pair=(
            _named_parameter_arrays(cuda_model),
            _named_parameter_arrays(cpu_model),
        ),
    )

    assert report.use_checkpoint is False
    assert report.nan_count == 0
    assert report.inf_count == 0
    over_tolerance = {
        name: getattr(report, name)
        for name in CUDA_SINGLE_STEP_TOLERANCES
        if getattr(report, name) is not None
        and getattr(report, name) > CUDA_SINGLE_STEP_TOLERANCES[name]
    }
    assert not over_tolerance, {
        "over_tolerance": over_tolerance,
        "tolerances": CUDA_SINGLE_STEP_TOLERANCES,
        "device_report": device_report.to_dict(),
        "report": report.to_dict(),
    }
