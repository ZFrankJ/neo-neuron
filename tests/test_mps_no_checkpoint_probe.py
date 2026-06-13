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

from src.runtime.parity_audit import BackendParityAuditReport, make_backend_parity_report
from src.runtime.backends import torch_backend
from src.train.optim import build_optimizer


requires_mps_probe = pytest.mark.skipif(
    os.environ.get("NEO_RUN_MPS_PROBE") != "1",
    reason="set NEO_RUN_MPS_PROBE=1 to run optional MPS diagnostics",
)

requires_mps_backend = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="PyTorch MPS backend is not available",
)


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


SINGLE_STEP_TOLERANCES = {
    "loss_diff": 5e-3,
    "logits_max_diff": 5e-3,
    "state_max_diff": 5e-3,
    "gradient_max_diff": 2e-2,
    "update_max_diff": 2e-2,
    "gradient_rel_l2": 5e-2,
}


SHORT_TRAJECTORY_STEPS = 8
SHORT_TRAJECTORY_TOLERANCES = {
    "initial_eval_loss_diff": 5e-3,
    "initial_eval_logits_max_diff": 5e-3,
    "initial_eval_state_max_diff": 5e-3,
    "max_step_loss_diff": 2e-2,
    "final_eval_loss_diff": 2e-2,
    "final_eval_logits_max_diff": 2e-2,
    "final_param_max_diff": 5e-2,
    "final_state_max_diff": 2e-2,
    "max_gradient_norm_diff": 5e-2,
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


def _trusted_mps_cfg(**overrides) -> dict:
    cfg = dict(TINY_NO_CHECKPOINT_CFG)
    cfg.update(overrides)
    if bool(cfg.get("use_checkpoint", False)):
        raise ValueError("trusted MPS parity probes require use_checkpoint=false")
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
    cfg = _trusted_mps_cfg() if cfg is None else _trusted_mps_cfg(**cfg)
    model = torch_backend.build_model(cfg, "neo").to(device)
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


def _shifted_tokens(device: torch.device, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, _ = _tokens(device)
    shift = int(step) * 3
    vocab = int(TINY_NO_CHECKPOINT_CFG["vocab_size"])
    return (x + shift) % vocab, (y + shift) % vocab


def _eval_outputs(
    model: torch.nn.Module,
    device: torch.device,
    state: torch.Tensor | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    x, y, base_state = _tokens(device)
    eval_state = base_state if state is None else state
    with torch.no_grad():
        logits, next_state = model(x, eval_state)
        loss = F.cross_entropy(logits.reshape(-1, int(TINY_NO_CHECKPOINT_CFG["vocab_size"])), y.reshape(-1))
    if device.type == "mps":
        _sync_mps()
    return (
        float(loss.detach().cpu().item()),
        logits.detach().cpu().numpy().copy(),
        next_state.detach().cpu().numpy().copy(),
    )


def _loss_grads_and_outputs(
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.train()
    x, y, state = _tokens(device)
    model.zero_grad(set_to_none=True)
    logits, next_state = model(x, state)
    loss = F.cross_entropy(logits.reshape(-1, int(TINY_NO_CHECKPOINT_CFG["vocab_size"])), y.reshape(-1))
    loss.backward()
    if device.type == "mps":
        _sync_mps()
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
    return {name: param.detach().cpu().numpy().copy() for name, param in model.named_parameters()}


def _gradient_l2_norm(values: dict[str, np.ndarray]) -> float:
    total = 0.0
    for value in values.values():
        finite = value[np.isfinite(value)]
        total += float(np.sum(finite * finite))
    return float(np.sqrt(total))


def _gradient_rel_l2(got: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> float:
    assert set(got) == set(expected)
    total_diff_sq = 0.0
    total_ref_sq = 0.0
    for name, expected_grad in expected.items():
        got_grad = got[name]
        assert np.all(np.isfinite(got_grad)), name
        diff = got_grad - expected_grad
        total_diff_sq += float(np.sum(diff * diff))
        total_ref_sq += float(np.sum(expected_grad * expected_grad))
    return float(np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30)))


def _single_step_cpu_mps_report() -> tuple[BackendParityAuditReport, float, str]:
    cfg = _trusted_mps_cfg()
    cpu_device = torch.device("cpu")
    mps_device = _mps_device()
    cpu_model = _build_model(cpu_device, cfg)
    mps_model = _build_model(mps_device, cfg)
    state = _deterministic_state_dict(cpu_model.state_dict())
    cpu_model.load_state_dict(state)
    mps_model.load_state_dict({name: value.to(mps_device) for name, value in state.items()})

    cpu_optimizer = build_optimizer(cpu_model, cfg)
    mps_optimizer = build_optimizer(mps_model, cfg)

    cpu_loss, cpu_logits, cpu_state, cpu_grads = _loss_grads_and_outputs(cpu_model, cpu_device)
    mps_loss, mps_logits, mps_state, mps_grads = _loss_grads_and_outputs(mps_model, mps_device)

    cpu_optimizer.step()
    mps_optimizer.step()
    _sync_mps()

    cpu_params = _named_parameter_arrays(cpu_model)
    mps_params = _named_parameter_arrays(mps_model)
    grad_rel_l2 = _gradient_rel_l2(mps_grads, cpu_grads)
    largest_update_name = max(
        cpu_params,
        key=lambda name: float(np.max(np.abs(mps_params[name] - cpu_params[name]))),
    )

    report = make_backend_parity_report(
        backend_pair=("torch", "torch"),
        device_pair=("cpu", "mps"),
        seed=int(cfg["seed"]),
        model_shape=_audit_model_shape(cfg),
        use_checkpoint=bool(cfg["use_checkpoint"]),
        loss_pair=(mps_loss, cpu_loss),
        logits_pair=(mps_logits, cpu_logits),
        state_pair=(mps_state, cpu_state),
        gradient_pair=(mps_grads, cpu_grads),
        update_pair=(mps_params, cpu_params),
    )
    return report, grad_rel_l2, largest_update_name


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    state: torch.Tensor,
    step: int,
) -> tuple[float, torch.Tensor, dict[str, np.ndarray], float]:
    model.train()
    x, y = _shifted_tokens(device, step)
    optimizer.zero_grad(set_to_none=True)
    logits, next_state = model(x, state)
    loss = F.cross_entropy(logits.reshape(-1, int(TINY_NO_CHECKPOINT_CFG["vocab_size"])), y.reshape(-1))
    loss.backward()
    if device.type == "mps":
        _sync_mps()
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        grads[name] = param.grad.detach().cpu().numpy().copy()
    grad_norm = _gradient_l2_norm(grads)
    optimizer.step()
    if device.type == "mps":
        _sync_mps()
    return float(loss.detach().cpu().item()), next_state.detach(), grads, grad_norm


def _max_named_array_diff(got: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> tuple[float, str]:
    assert set(got) == set(expected)
    largest_name = ""
    largest_diff = 0.0
    for name in sorted(got):
        diff = float(np.max(np.abs(got[name] - expected[name])))
        if diff >= largest_diff:
            largest_name = name
            largest_diff = diff
    return largest_diff, largest_name


def _nonfinite_count(*arrays: np.ndarray, named_arrays: dict[str, np.ndarray] | None = None) -> int:
    count = 0
    for array in arrays:
        count += int(np.isnan(array).sum() + np.isinf(array).sum())
    if named_arrays is not None:
        for array in named_arrays.values():
            count += int(np.isnan(array).sum() + np.isinf(array).sum())
    return count


def _short_trajectory_cpu_mps_report() -> dict[str, float | int | str | dict]:
    cfg = _trusted_mps_cfg()
    cpu_device = torch.device("cpu")
    mps_device = _mps_device()
    cpu_model = _build_model(cpu_device, cfg)
    mps_model = _build_model(mps_device, cfg)
    state = _deterministic_state_dict(cpu_model.state_dict())
    cpu_model.load_state_dict(state)
    mps_model.load_state_dict({name: value.to(mps_device) for name, value in state.items()})

    cpu_initial_loss, cpu_initial_logits, cpu_initial_eval_state = _eval_outputs(cpu_model, cpu_device)
    mps_initial_loss, mps_initial_logits, mps_initial_eval_state = _eval_outputs(mps_model, mps_device)
    initial_report = make_backend_parity_report(
        backend_pair=("torch", "torch"),
        device_pair=("cpu", "mps"),
        seed=int(cfg["seed"]),
        model_shape=_audit_model_shape(cfg),
        use_checkpoint=bool(cfg["use_checkpoint"]),
        loss_pair=(cpu_initial_loss, mps_initial_loss),
        logits_pair=(cpu_initial_logits, mps_initial_logits),
        state_pair=(cpu_initial_eval_state, mps_initial_eval_state),
    )

    cpu_optimizer = build_optimizer(cpu_model, cfg)
    mps_optimizer = build_optimizer(mps_model, cfg)
    cpu_state = _tokens(cpu_device)[2]
    mps_state = _tokens(mps_device)[2]
    step_loss_diffs = []
    gradient_norm_diffs = []
    nonfinite_count = initial_report.nan_count + initial_report.inf_count

    for step in range(SHORT_TRAJECTORY_STEPS):
        cpu_loss, cpu_state, cpu_grads, cpu_grad_norm = _train_step(
            cpu_model,
            cpu_optimizer,
            cpu_device,
            cpu_state,
            step,
        )
        mps_loss, mps_state, mps_grads, mps_grad_norm = _train_step(
            mps_model,
            mps_optimizer,
            mps_device,
            mps_state,
            step,
        )
        step_loss_diffs.append(abs(cpu_loss - mps_loss))
        gradient_norm_diffs.append(abs(cpu_grad_norm - mps_grad_norm))
        nonfinite_count += _nonfinite_count(
            np.asarray([cpu_loss, mps_loss, cpu_grad_norm, mps_grad_norm], dtype=np.float64),
            named_arrays=cpu_grads,
        )
        nonfinite_count += _nonfinite_count(named_arrays=mps_grads)

    cpu_final_loss, cpu_final_logits, _ = _eval_outputs(cpu_model, cpu_device)
    mps_final_loss, mps_final_logits, _ = _eval_outputs(mps_model, mps_device)
    cpu_params = _named_parameter_arrays(cpu_model)
    mps_params = _named_parameter_arrays(mps_model)
    final_param_max_diff, largest_param = _max_named_array_diff(mps_params, cpu_params)
    final_state_max_diff = float(np.max(np.abs(mps_state.detach().cpu().numpy() - cpu_state.detach().cpu().numpy())))

    final_report = make_backend_parity_report(
        backend_pair=("torch", "torch"),
        device_pair=("cpu", "mps"),
        seed=int(cfg["seed"]),
        model_shape=_audit_model_shape(cfg),
        use_checkpoint=bool(cfg["use_checkpoint"]),
        loss_pair=(cpu_final_loss, mps_final_loss),
        logits_pair=(cpu_final_logits, mps_final_logits),
        state_pair=(cpu_state.detach().cpu().numpy(), mps_state.detach().cpu().numpy()),
        update_pair=(cpu_params, mps_params),
    )
    nonfinite_count += final_report.nan_count + final_report.inf_count

    return {
        "steps": SHORT_TRAJECTORY_STEPS,
        "initial_eval_loss_diff": initial_report.loss_diff,
        "initial_eval_logits_max_diff": initial_report.logits_max_diff,
        "initial_eval_state_max_diff": initial_report.state_max_diff,
        "max_step_loss_diff": max(step_loss_diffs),
        "final_eval_loss_diff": final_report.loss_diff,
        "final_eval_logits_max_diff": final_report.logits_max_diff,
        "final_param_max_diff": final_param_max_diff,
        "final_state_max_diff": final_state_max_diff,
        "max_gradient_norm_diff": max(gradient_norm_diffs),
        "nonfinite_count": nonfinite_count,
        "largest_parameter": largest_param,
        "initial_report": initial_report.to_dict(),
        "final_report": final_report.to_dict(),
    }


@requires_mps_probe
@requires_mps_backend
def test_optional_no_checkpoint_mps_gradients_match_cpu_tiny_probe():
    cpu_model = _build_model(torch.device("cpu"))
    mps_model = _build_model(_mps_device())
    state = _deterministic_state_dict(cpu_model.state_dict())
    cpu_model.load_state_dict(state)
    mps_model.load_state_dict({name: value.to(_mps_device()) for name, value in state.items()})

    cpu_loss, _, _, cpu_grads = _loss_grads_and_outputs(cpu_model, torch.device("cpu"))
    mps_loss, _, _, mps_grads = _loss_grads_and_outputs(mps_model, _mps_device())

    assert abs(cpu_loss - mps_loss) <= 5e-3
    max_diff = 0.0
    assert set(mps_grads) == set(cpu_grads)
    for name, cpu_grad in cpu_grads.items():
        mps_grad = mps_grads[name]
        assert np.all(np.isfinite(mps_grad)), name
        diff = mps_grad - cpu_grad
        max_diff = max(max_diff, float(np.max(np.abs(diff))))
    rel_l2 = _gradient_rel_l2(mps_grads, cpu_grads)
    assert rel_l2 <= 5e-2
    assert max_diff <= 2e-2


@requires_mps_probe
@requires_mps_backend
def test_optional_no_checkpoint_mps_single_step_parity_envelope():
    report, grad_rel_l2, largest_update_name = _single_step_cpu_mps_report()

    assert report.use_checkpoint is False
    assert report.nan_count == 0
    assert report.inf_count == 0
    assert report.loss_diff is not None
    assert report.logits_max_diff is not None
    assert report.state_max_diff is not None
    assert report.gradient_max_diff is not None
    assert report.update_max_diff is not None

    failures = {
        "loss_diff": report.loss_diff,
        "logits_max_diff": report.logits_max_diff,
        "state_max_diff": report.state_max_diff,
        "gradient_max_diff": report.gradient_max_diff,
        "update_max_diff": report.update_max_diff,
        "gradient_rel_l2": grad_rel_l2,
    }
    over_tolerance = {
        name: value
        for name, value in failures.items()
        if value > SINGLE_STEP_TOLERANCES[name]
    }
    assert not over_tolerance, {
        "over_tolerance": over_tolerance,
        "tolerances": SINGLE_STEP_TOLERANCES,
        "largest_update_parameter": largest_update_name,
        "report": report.to_dict(),
    }


def test_trusted_mps_probe_rejects_checkpointed_path():
    with pytest.raises(ValueError, match="use_checkpoint=false"):
        _trusted_mps_cfg(use_checkpoint=True)


@requires_mps_probe
@requires_mps_backend
def test_optional_no_checkpoint_mps_short_training_trajectory_stays_inside_envelope():
    trajectory = _short_trajectory_cpu_mps_report()

    assert trajectory["nonfinite_count"] == 0, trajectory
    over_tolerance = {
        name: trajectory[name]
        for name in SHORT_TRAJECTORY_TOLERANCES
        if trajectory[name] > SHORT_TRAJECTORY_TOLERANCES[name]
    }
    assert not over_tolerance, {
        "over_tolerance": over_tolerance,
        "tolerances": SHORT_TRAJECTORY_TOLERANCES,
        "largest_parameter": trajectory["largest_parameter"],
        "trajectory": trajectory,
    }


@requires_mps_probe
@requires_mps_backend
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
