"""Deterministic MLX-reference LSTM training parity tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

mx = pytest.importorskip("mlx.core")
mxnn = pytest.importorskip("mlx.nn")
mxoptim = pytest.importorskip("mlx.optimizers")
mlx_utils = pytest.importorskip("mlx.utils")

from src.runtime.backends import mlx_backend, torch_backend
from src.runtime.checkpoint_compat import map_model_state, to_numpy_state_dict
from src.train.optim import build_optimizer


LOSS_ATOL = 1e-6
GRADIENT_ATOL = 1e-5
UPDATE_ATOL = 1e-5
TRAJECTORY_ATOL = 2e-5
TRAJECTORY_STEPS = 12


def _cfg(**overrides):
    cfg = {
        "model_name": "lstm",
        "vocab_size": 23,
        "d_model": 5,
        "d_embed": 3,
        "n_layers": 2,
        "dropout": 0.0,
        "tie_embeddings": True,
        "recurrent_norm": "rmsnorm",
        "recurrent_norm_place": "all",
        "rmsnorm_eps": 1e-5,
        "lstm_bias_mode": "single",
        "reference_backend": "mlx",
        "use_checkpoint": False,
        "weight_decay_policy": "table",
        "embed_weight_decay": 1e-4,
        "proj_weight_decay": 1e-3,
        "recurrent_weight_decay": 2e-4,
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "adam_eps": 1e-8,
    }
    cfg.update(overrides)
    return cfg


def _batch(step: int = 0):
    x = (np.arange(12, dtype=np.int32).reshape(4, 3) * 5 + 1 + step * 3) % 23
    y = (x + 4) % 23
    return x, y.astype(np.int32)


def _initial_state(cfg):
    rng = np.random.default_rng(20260714)
    shape = (int(cfg["n_layers"]), 3, int(cfg["d_model"]))
    return (
        rng.normal(0.0, 0.02, size=shape).astype(np.float32),
        rng.normal(0.0, 0.02, size=shape).astype(np.float32),
    )


def _mlx_params(model):
    return {
        name: np.asarray(value)
        for name, value in mlx_utils.tree_flatten(model.parameters())
    }


def _build_mapped_pair(cfg):
    mlx_model = mlx_backend.build_model(cfg, "lstm")
    torch_model = torch_backend.build_model(cfg, "lstm")
    mapped, warnings = map_model_state(
        model_name="lstm",
        src_backend="mlx",
        dst_backend="torch",
        src_state_np=_mlx_params(mlx_model),
        dst_template=torch_model.state_dict(),
        cfg=cfg,
    )
    assert warnings == []
    torch_model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in mapped.items()}
    )
    return torch_model, mlx_model


def _torch_step_data(model, cfg, x_np, y_np, state_np):
    model.train()
    model.zero_grad(set_to_none=True)
    logits, state = model(
        torch.from_numpy(x_np).long(),
        tuple(torch.from_numpy(value) for value in state_np),
    )
    loss = F.cross_entropy(
        logits.reshape(-1, int(cfg["vocab_size"])),
        torch.from_numpy(y_np).reshape(-1).long(),
    )
    loss.backward()
    grads = {
        name: parameter.grad.detach().numpy().copy()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return (
        float(loss.detach()),
        grads,
        tuple(value.detach().numpy().copy() for value in state),
    )


def _mlx_loss_and_grad(model, cfg):
    def loss_fn(x, y, state):
        logits, next_state = model(x, state)
        loss = mx.mean(
            mxnn.losses.cross_entropy(
                logits.reshape(-1, int(cfg["vocab_size"])),
                y.reshape(-1),
            )
        )
        return loss, next_state

    return mxnn.value_and_grad(model, loss_fn)


def _mlx_step_data(loss_and_grad, x_np, y_np, state_np):
    (loss, state), grads = loss_and_grad(
        mx.array(x_np),
        mx.array(y_np),
        tuple(mx.array(value) for value in state_np),
    )
    mx.eval(loss, state, grads)
    return (
        float(loss.item()),
        {
            name: np.asarray(value)
            for name, value in mlx_utils.tree_flatten(grads)
        },
        tuple(np.asarray(value) for value in state),
        grads,
    )


def _map_mlx_to_torch(values, torch_model, cfg):
    mapped, warnings = map_model_state(
        model_name="lstm",
        src_backend="mlx",
        dst_backend="torch",
        src_state_np=values,
        dst_template=torch_model.state_dict(),
        cfg=cfg,
    )
    assert warnings == []
    return {
        name: value
        for name, value in mapped.items()
        if dict(torch_model.named_parameters()).get(name) is not None
        and dict(torch_model.named_parameters())[name].requires_grad
    }


def _tree_l2(values):
    return math.sqrt(sum(float(np.sum(value * value)) for value in values.values()))


def _max_tree_diff(left, right):
    return max(float(np.max(np.abs(left[name] - right[name]))) for name in left)


def _mlx_optimizer(model, cfg):
    return mxoptim.AdamW(
        learning_rate=float(cfg["lr"]),
        betas=list(cfg["betas"]),
        eps=float(cfg["adam_eps"]),
        weight_decay=0.0,
    )


def _mlx_optimizer_state(optimizer):
    return {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(optimizer.state)
    }


def _mlx_update(model, optimizer, grads, cfg):
    optimizer.update(model, grads)
    mlx_backend._apply_decoupled_weight_decay(
        model,
        mlx_backend._build_weight_decay_lookup(model, "lstm", cfg),
        float(cfg["lr"]),
    )
    mx.eval(model.parameters(), optimizer.state)


def _torch_decay_by_name(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        names[id(parameter)]: float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_lstm_loss_every_mapped_gradient_norm_and_recurrent_state_match():
    cfg = _cfg()
    assert cfg["lstm_bias_mode"] == "single"
    assert cfg["rmsnorm_eps"] == pytest.approx(1e-5)
    assert cfg["reference_backend"] == "mlx"
    assert cfg["dropout"] == 0.0
    assert cfg["use_checkpoint"] is False

    torch_model, mlx_model = _build_mapped_pair(cfg)
    x_np, y_np = _batch()
    state_np = _initial_state(cfg)
    torch_loss, torch_grads, torch_state = _torch_step_data(
        torch_model, cfg, x_np, y_np, state_np
    )
    mlx_loss, mlx_grads, mlx_state, _ = _mlx_step_data(
        _mlx_loss_and_grad(mlx_model, cfg), x_np, y_np, state_np
    )
    mapped_grads = _map_mlx_to_torch(mlx_grads, torch_model, cfg)

    assert set(torch_grads) == set(mapped_grads)
    assert abs(torch_loss - mlx_loss) <= LOSS_ATOL
    assert _max_tree_diff(torch_grads, mapped_grads) <= GRADIENT_ATOL
    assert abs(_tree_l2(torch_grads) - _tree_l2(mapped_grads)) <= GRADIENT_ATOL
    np.testing.assert_allclose(torch_state[0], mlx_state[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(torch_state[1], mlx_state[1], rtol=1e-5, atol=1e-6)
    assert all(np.isfinite(value).all() for value in torch_grads.values())
    assert all(np.isfinite(value).all() for value in mlx_grads.values())


def test_lstm_public_optimizer_roles_and_one_update_match_mlx():
    cfg = _cfg()
    torch_model, mlx_model = _build_mapped_pair(cfg)
    torch_optimizer = build_optimizer(torch_model, cfg)
    torch_decay = _torch_decay_by_name(torch_model, torch_optimizer)
    mlx_decay = mlx_backend._build_weight_decay_lookup(mlx_model, "lstm", cfg)

    assert torch_decay["emb.weight"] == pytest.approx(cfg["embed_weight_decay"])
    assert torch_decay["in_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert torch_decay["out_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert torch_decay["lstm.weight_ih_l0"] == pytest.approx(
        cfg["recurrent_weight_decay"]
    )
    assert torch_decay["lstm.weight_hh_l1"] == pytest.approx(
        cfg["recurrent_weight_decay"]
    )
    assert torch_decay["lstm.pre_norms.0.weight"] == pytest.approx(0.0)
    assert torch_decay["lstm.stack_norm.weight"] == pytest.approx(0.0)
    assert sorted(torch_decay.values()) == pytest.approx(sorted(mlx_decay.values()))

    x_np, y_np = _batch()
    state_np = _initial_state(cfg)
    _, _, _ = _torch_step_data(torch_model, cfg, x_np, y_np, state_np)
    _, _, _, mlx_grads = _mlx_step_data(
        _mlx_loss_and_grad(mlx_model, cfg), x_np, y_np, state_np
    )
    torch_optimizer.step()
    mlx_optimizer = _mlx_optimizer(mlx_model, cfg)
    _mlx_update(mlx_model, mlx_optimizer, mlx_grads, cfg)

    mapped_after = _map_mlx_to_torch(_mlx_params(mlx_model), torch_model, cfg)
    torch_after = {
        name: parameter.detach().numpy()
        for name, parameter in torch_model.named_parameters()
        if parameter.requires_grad
    }
    assert _max_tree_diff(torch_after, mapped_after) <= UPDATE_ATOL


def test_lstm_single_bias_update_matches_and_split_bias_would_fail_parity():
    cfg = _cfg()
    x_np, y_np = _batch()
    state_np = _initial_state(cfg)
    single_model, mlx_model = _build_mapped_pair(cfg)
    split_model = torch_backend.build_model(_cfg(lstm_bias_mode="split"), "lstm")
    split_model.load_state_dict(single_model.state_dict())

    single_optimizer = build_optimizer(single_model, cfg)
    split_optimizer = build_optimizer(split_model, _cfg(lstm_bias_mode="split"))
    _torch_step_data(single_model, cfg, x_np, y_np, state_np)
    _torch_step_data(split_model, cfg, x_np, y_np, state_np)
    _, _, _, mlx_grads = _mlx_step_data(
        _mlx_loss_and_grad(mlx_model, cfg), x_np, y_np, state_np
    )
    single_optimizer.step()
    split_optimizer.step()
    mlx_optimizer = _mlx_optimizer(mlx_model, cfg)
    _mlx_update(mlx_model, mlx_optimizer, mlx_grads, cfg)

    mlx_bias = _mlx_params(mlx_model)["lstm_layers.0.bias"]
    single_effective = (
        single_model.lstm.bias_ih_l0 + single_model.lstm.bias_hh_l0
    ).detach().numpy()
    split_effective = (
        split_model.lstm.bias_ih_l0 + split_model.lstm.bias_hh_l0
    ).detach().numpy()
    assert float(np.max(np.abs(single_effective - mlx_bias))) <= UPDATE_ATOL
    assert float(np.max(np.abs(split_effective - mlx_bias))) > UPDATE_ATOL


def test_lstm_fixed_batch_short_training_trajectory_stays_inside_envelope():
    cfg = _cfg()
    torch_model, mlx_model = _build_mapped_pair(cfg)
    torch_optimizer = build_optimizer(torch_model, cfg)
    mlx_optimizer = _mlx_optimizer(mlx_model, cfg)
    mlx_loss_and_grad = _mlx_loss_and_grad(mlx_model, cfg)
    torch_state = _initial_state(cfg)
    mlx_state = tuple(value.copy() for value in torch_state)
    max_loss_diff = 0.0
    max_gradient_norm_diff = 0.0
    nonfinite_count = 0

    for step in range(TRAJECTORY_STEPS):
        x_np, y_np = _batch(step)
        torch_loss, torch_grads, torch_state_out = _torch_step_data(
            torch_model, cfg, x_np, y_np, torch_state
        )
        mlx_loss, mlx_grads, mlx_state_out, mlx_grads_tree = _mlx_step_data(
            mlx_loss_and_grad, x_np, y_np, mlx_state
        )
        mapped_grads = _map_mlx_to_torch(mlx_grads, torch_model, cfg)
        max_loss_diff = max(max_loss_diff, abs(torch_loss - mlx_loss))
        max_gradient_norm_diff = max(
            max_gradient_norm_diff,
            abs(_tree_l2(torch_grads) - _tree_l2(mapped_grads)),
        )
        nonfinite_count += sum(
            int(not np.isfinite(value).all())
            for value in (*torch_grads.values(), *mlx_grads.values())
        )
        torch_optimizer.step()
        _mlx_update(mlx_model, mlx_optimizer, mlx_grads_tree, cfg)
        torch_state = tuple(value.copy() for value in torch_state_out)
        mlx_state = tuple(value.copy() for value in mlx_state_out)

    mapped_after = _map_mlx_to_torch(_mlx_params(mlx_model), torch_model, cfg)
    torch_after = {
        name: parameter.detach().numpy()
        for name, parameter in torch_model.named_parameters()
        if parameter.requires_grad
    }
    assert nonfinite_count == 0
    assert max_loss_diff <= TRAJECTORY_ATOL
    assert max_gradient_norm_diff <= TRAJECTORY_ATOL
    assert _max_tree_diff(torch_after, mapped_after) <= TRAJECTORY_ATOL
    np.testing.assert_allclose(torch_state[0], mlx_state[0], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(torch_state[1], mlx_state[1], rtol=2e-5, atol=2e-6)


def _torch_training_step(model, optimizer, cfg, step):
    x_np, y_np = _batch(step)
    state_np = _initial_state(cfg)
    _torch_step_data(model, cfg, x_np, y_np, state_np)
    optimizer.step()


def _mlx_training_step(model, optimizer, cfg, step):
    x_np, y_np = _batch(step)
    state_np = _initial_state(cfg)
    _, _, _, grads = _mlx_step_data(
        _mlx_loss_and_grad(model, cfg), x_np, y_np, state_np
    )
    _mlx_update(model, optimizer, grads, cfg)


def test_lstm_same_backend_optimizer_resume_is_exact(tmp_path: Path):
    cfg = _cfg()

    torch_model = torch_backend.build_model(cfg, "lstm")
    torch_optimizer = build_optimizer(torch_model, cfg)
    _torch_training_step(torch_model, torch_optimizer, cfg, 0)
    torch_path = tmp_path / "torch-resume.pkl"
    torch_backend.save_checkpoint_entry(
        torch_path, torch_model, torch_optimizer, None, 1, 1, cfg
    )
    _torch_training_step(torch_model, torch_optimizer, cfg, 1)

    resumed_torch = torch_backend.build_model(cfg, "lstm")
    resumed_torch_optimizer = build_optimizer(resumed_torch, cfg)
    torch_backend.load_checkpoint_entry(
        torch_path,
        resumed_torch,
        optimizer=resumed_torch_optimizer,
        device=torch.device("cpu"),
        cfg=cfg,
    )
    _torch_training_step(resumed_torch, resumed_torch_optimizer, cfg, 1)
    for name, value in torch_model.state_dict().items():
        torch.testing.assert_close(value, resumed_torch.state_dict()[name], rtol=0, atol=0)

    mlx_model = mlx_backend.build_model(cfg, "lstm")
    mlx_optimizer = _mlx_optimizer(mlx_model, cfg)
    _mlx_training_step(mlx_model, mlx_optimizer, cfg, 0)
    mlx_path = tmp_path / "mlx-resume.pkl"
    mlx_backend.save_checkpoint_entry(
        mlx_path, mlx_model, mlx_optimizer, None, 1, 1, cfg
    )
    _mlx_training_step(mlx_model, mlx_optimizer, cfg, 1)

    resumed_mlx = mlx_backend.build_model(cfg, "lstm")
    resumed_mlx_optimizer = _mlx_optimizer(resumed_mlx, cfg)
    mlx_backend.load_checkpoint_entry(
        mlx_path, resumed_mlx, optimizer=resumed_mlx_optimizer, cfg=cfg
    )
    _mlx_training_step(resumed_mlx, resumed_mlx_optimizer, cfg, 1)
    assert _max_tree_diff(_mlx_params(mlx_model), _mlx_params(resumed_mlx)) == 0.0


def test_cross_backend_optimizer_resume_warns_and_does_not_map_state(tmp_path: Path):
    cfg = _cfg()
    mlx_model = mlx_backend.build_model(cfg, "lstm")
    mlx_optimizer = _mlx_optimizer(mlx_model, cfg)
    _mlx_training_step(mlx_model, mlx_optimizer, cfg, 0)
    path = tmp_path / "mlx-with-optimizer.pkl"
    mlx_backend.save_checkpoint_entry(path, mlx_model, mlx_optimizer, None, 1, 1, cfg)

    torch_model = torch_backend.build_model(cfg, "lstm")
    torch_optimizer = build_optimizer(torch_model, cfg)
    assert torch_optimizer.state == {}
    with pytest.warns(UserWarning, match="cross-backend optimizer resume"):
        torch_backend.load_checkpoint_entry(
            path,
            torch_model,
            optimizer=torch_optimizer,
            device=torch.device("cpu"),
            cfg=cfg,
        )
    assert torch_optimizer.state == {}

    torch_path = tmp_path / "torch-with-optimizer.pkl"
    _torch_training_step(torch_model, torch_optimizer, cfg, 0)
    torch_backend.save_checkpoint_entry(
        torch_path, torch_model, torch_optimizer, None, 1, 1, cfg
    )
    resumed_mlx = mlx_backend.build_model(cfg, "lstm")
    resumed_mlx_optimizer = _mlx_optimizer(resumed_mlx, cfg)
    initial_mlx_optimizer_state = _mlx_optimizer_state(resumed_mlx_optimizer)
    with pytest.warns(UserWarning, match="cross-backend optimizer resume"):
        mlx_backend.load_checkpoint_entry(
            torch_path, resumed_mlx, optimizer=resumed_mlx_optimizer, cfg=cfg
        )
    restored_mlx_optimizer_state = _mlx_optimizer_state(resumed_mlx_optimizer)
    assert set(restored_mlx_optimizer_state) == set(initial_mlx_optimizer_state)
    for name, value in initial_mlx_optimizer_state.items():
        np.testing.assert_array_equal(restored_mlx_optimizer_state[name], value)
