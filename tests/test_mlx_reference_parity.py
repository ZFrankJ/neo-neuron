"""Small MLX-reference parity tests for the Neo PyTorch backend."""

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


CANONICAL_PARITY_CFG = {
    "model_name": "neo",
    "vocab_size": 2048,
    "d_model": 128,
    "d_embed": 64,
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
    "block_size": 64,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay_policy": "table",
    "seed": 42,
    "reference_backend": "mlx",
}


def _tiny_cfg(**overrides):
    cfg = dict(CANONICAL_PARITY_CFG)
    cfg.update(
        {
            "vocab_size": 64,
            "d_model": 16,
            "d_embed": 8,
            "block_size": 8,
            "batch_size": 3,
            "proj_weight_decay": 0.0,
            "recurrent_weight_decay": 0.0,
            "embed_weight_decay": 0.0,
            "betas": (0.0, 0.0),
            "adam_eps": 1e-8,
        }
    )
    cfg.update(overrides)
    return cfg


def _production_like_optimizer_cfg(**overrides):
    cfg = _tiny_cfg(
        vocab_size=32,
        d_model=8,
        d_embed=4,
        n_layers=2,
        block_size=6,
        batch_size=2,
        recurrent_norm="rmsnorm",
        activation_id="id5",
        use_checkpoint=False,
        betas=(0.9, 0.95),
        adam_eps=1e-8,
        embed_weight_decay=1e-4,
        proj_weight_decay=1e-3,
        recurrent_weight_decay=2e-4,
        weight_decay_policy="table",
        grad_clip=0.0,
    )
    cfg.update(overrides)
    return cfg


def _build_pair(cfg):
    torch_model = torch_backend.build_model(cfg, "neo").to("cpu")
    mlx_model = mlx_backend.build_model(cfg, "neo")
    return torch_model, mlx_model


def _mlx_params(model):
    return dict(mlx_utils.tree_flatten(model.parameters()))


def _deterministic_state(template):
    rng = np.random.default_rng(1234)
    state = {}
    for name, tensor in template.items():
        values = rng.normal(0.0, 0.03, size=tuple(tensor.shape)).astype(np.float32)
        if name.endswith("output_bias") or name.endswith(".bias"):
            values.fill(0.0)
        if "norm" in name and name.endswith(".weight"):
            values = 1.0 + rng.normal(0.0, 0.01, size=tuple(tensor.shape)).astype(np.float32)
        state[name] = values
    return state


def _load_same_state(torch_model, mlx_model, state):
    torch_state = {name: torch.from_numpy(value.copy()) for name, value in state.items()}
    torch_model.load_state_dict(torch_state)
    mlx_model.update(mlx_utils.tree_unflatten([(name, mx.array(value)) for name, value in state.items()]))
    mx.eval(mlx_model.parameters())


def _tokens(cfg):
    t = int(cfg["block_size"])
    b = int(cfg["batch_size"])
    vocab = int(cfg["vocab_size"])
    x = (np.arange(t * b, dtype=np.int64).reshape(t, b) * 7 + 3) % vocab
    y = (x + 5) % vocab
    return x, y


def _state(cfg):
    rng = np.random.default_rng(5678)
    return rng.normal(
        0.0,
        0.02,
        size=(int(cfg["n_layers"]), int(cfg["batch_size"]), int(cfg["d_model"])),
    ).astype(np.float32)


def _forward_pair(torch_model, mlx_model, cfg):
    x_np, _ = _tokens(cfg)
    state_np = _state(cfg)
    x_torch = torch.from_numpy(x_np).long()
    state_torch = torch.from_numpy(state_np)
    x_mlx = mx.array(x_np.astype(np.int32))
    state_mlx = mx.array(state_np)

    with torch.no_grad():
        torch_logits, torch_state = torch_model(x_torch, state_torch)
    mlx_logits, mlx_state = mlx_model(x_mlx, state_mlx)
    mx.eval(mlx_logits, mlx_state)
    return (
        torch_logits.detach().numpy(),
        np.asarray(mlx_logits),
        torch_state.detach().numpy(),
        np.asarray(mlx_state),
    )


def _torch_loss_and_grads(model, cfg):
    model.train()
    x_np, y_np = _tokens(cfg)
    state_np = _state(cfg)
    x = torch.from_numpy(x_np).long()
    y = torch.from_numpy(y_np).long()
    state = torch.from_numpy(state_np)
    model.zero_grad(set_to_none=True)
    logits, _ = model(x, state)
    loss = F.cross_entropy(logits.reshape(-1, int(cfg["vocab_size"])), y.reshape(-1))
    loss.backward()
    grads = {name: param.grad.detach().numpy().copy() for name, param in model.named_parameters()}
    return float(loss.detach().item()), grads


def _mlx_loss_and_grads(model, cfg):
    model.train()
    x_np, y_np = _tokens(cfg)
    state_np = _state(cfg)
    x = mx.array(x_np.astype(np.int32))
    y = mx.array(y_np.astype(np.int32))
    state = mx.array(state_np)

    def loss_fn(batch_x, batch_y, batch_state):
        logits, _ = model(batch_x, batch_state)
        return mx.mean(mxnn.losses.cross_entropy(logits.reshape(-1, int(cfg["vocab_size"])), batch_y.reshape(-1)))

    loss, grads = mxnn.value_and_grad(model, loss_fn)(x, y, state)
    mx.eval(loss, grads)
    return float(loss.item()), {name: np.asarray(value) for name, value in mlx_utils.tree_flatten(grads)}


def _assert_close(name, got, expected, atol=1e-6):
    diff = float(np.max(np.abs(got - expected))) if got.size else 0.0
    assert diff <= atol, f"{name} max_abs_diff={diff}"


def _l2_norm_tree(state):
    return math.sqrt(sum(float(np.sum(value * value)) for value in state.values()))


def _torch_weight_decay_by_name(model, optimizer):
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    out = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            out[id_to_name[id(param)]] = float(group.get("weight_decay", 0.0))
    return out


def test_canonical_tiny_config_constructs_and_maps_parameters():
    torch_model, mlx_model = _build_pair(CANONICAL_PARITY_CFG)
    torch_template = torch_model.state_dict()
    mlx_template = _mlx_params(mlx_model)

    assert CANONICAL_PARITY_CFG["use_checkpoint"] is False
    assert torch_model.recurrent.pre_norms[0].eps == pytest.approx(1e-5)
    assert set(torch_template) == set(mlx_template)
    for name, tensor in torch_template.items():
        assert tuple(tensor.shape) == tuple(mlx_template[name].shape)

    mapped, warnings = map_model_state(
        model_name="neo",
        src_backend="mlx",
        dst_backend="torch",
        src_state_np={name: np.asarray(value) for name, value in mlx_template.items()},
        dst_template=torch_template,
        cfg=CANONICAL_PARITY_CFG,
    )
    assert warnings == []
    assert set(mapped) == set(torch_template)


@pytest.mark.parametrize("recurrent_norm", ["none", "layernorm", "rmsnorm"])
@pytest.mark.parametrize("activation_id", ["id4", "id5"])
@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_forward_and_recurrent_state_match_mlx_reference(recurrent_norm, activation_id, n_layers):
    cfg = _tiny_cfg(recurrent_norm=recurrent_norm, activation_id=activation_id, n_layers=n_layers)
    torch_model, mlx_model = _build_pair(cfg)
    state = _deterministic_state(torch_model.state_dict())
    _load_same_state(torch_model, mlx_model, state)

    torch_model.eval()
    mlx_model.eval()
    torch_logits, mlx_logits, torch_state, mlx_state = _forward_pair(torch_model, mlx_model, cfg)

    _assert_close("logits", torch_logits, mlx_logits)
    _assert_close("state", torch_state, mlx_state)


def test_gradient_parity_uses_no_checkpoint_path():
    cfg = _tiny_cfg(recurrent_norm="rmsnorm", activation_id="id5", n_layers=2, use_checkpoint=False)
    torch_model, mlx_model = _build_pair(cfg)
    state = _deterministic_state(torch_model.state_dict())
    _load_same_state(torch_model, mlx_model, state)

    torch_loss, torch_grads = _torch_loss_and_grads(torch_model, cfg)
    mlx_loss, mlx_grads = _mlx_loss_and_grads(mlx_model, cfg)

    assert abs(torch_loss - mlx_loss) <= 1e-6
    total_diff_sq = 0.0
    total_ref_sq = 0.0
    max_diff = 0.0
    assert set(torch_grads) == set(mlx_grads)
    for name, torch_grad in torch_grads.items():
        mlx_grad = mlx_grads[name]
        diff = torch_grad - mlx_grad
        max_diff = max(max_diff, float(np.max(np.abs(diff))))
        total_diff_sq += float(np.sum(diff * diff))
        total_ref_sq += float(np.sum(mlx_grad * mlx_grad))
    rel_l2 = math.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))
    assert rel_l2 <= 1e-5
    assert max_diff <= 1e-5


def test_one_step_optimizer_parity_with_explicit_tiny_contract():
    cfg = _tiny_cfg(recurrent_norm="rmsnorm", activation_id="id5", n_layers=2)
    torch_model, mlx_model = _build_pair(cfg)
    state = _deterministic_state(torch_model.state_dict())
    _load_same_state(torch_model, mlx_model, state)
    _, torch_grads = _torch_loss_and_grads(torch_model, cfg)
    _, mlx_grads = _mlx_loss_and_grads(mlx_model, cfg)

    optimizer = build_optimizer(torch_model, cfg)
    for name, param in torch_model.named_parameters():
        param.grad = torch.from_numpy(torch_grads[name].copy())
    optimizer.step()

    mlx_optimizer = mxoptim.AdamW(
        learning_rate=float(cfg["lr"]),
        betas=list(cfg["betas"]),
        eps=float(cfg["adam_eps"]),
        weight_decay=0.0,
    )
    mlx_optimizer.update(mlx_model, mlx_utils.tree_unflatten([(name, mx.array(value)) for name, value in mlx_grads.items()]))
    mx.eval(mlx_model.parameters(), mlx_optimizer.state)

    torch_after = to_numpy_state_dict(torch_model.state_dict())
    mlx_after = {name: np.asarray(value) for name, value in _mlx_params(mlx_model).items()}
    for name, torch_value in torch_after.items():
        _assert_close(name, torch_value, mlx_after[name], atol=1e-5)


def test_production_like_optimizer_weight_decay_policy_matches_mlx_reference():
    cfg = _production_like_optimizer_cfg()
    torch_model, mlx_model = _build_pair(cfg)

    torch_optimizer = build_optimizer(torch_model, cfg)
    torch_wd = _torch_weight_decay_by_name(torch_model, torch_optimizer)
    mlx_wd = mlx_backend._build_weight_decay_lookup(mlx_model, "neo", cfg)

    assert torch_wd == pytest.approx(mlx_wd)
    assert torch_wd["emb.weight"] == pytest.approx(cfg["embed_weight_decay"])
    assert torch_wd["in_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert torch_wd["out_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert torch_wd["recurrent.layers.0.fg_linear.weight"] == pytest.approx(cfg["recurrent_weight_decay"])
    assert torch_wd["recurrent.layers.1.fg_linear.weight"] == pytest.approx(cfg["recurrent_weight_decay"])
    assert torch_wd["recurrent.pre_norms.0.weight"] == pytest.approx(cfg["recurrent_weight_decay"])
    assert torch_wd["recurrent.pre_norms.1.weight"] == pytest.approx(cfg["recurrent_weight_decay"])
    for name in (
        "output_bias",
        "in_proj.bias",
        "out_proj.bias",
        "recurrent.layers.0.fg_linear.bias",
        "recurrent.layers.1.fg_linear.bias",
        "recurrent.stack_norm.weight",
    ):
        assert torch_wd[name] == pytest.approx(0.0)
        assert mlx_wd[name] == pytest.approx(0.0)


def test_mlx_reference_optimizer_applies_decoupled_decay_after_adam_update():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    cfg = {
        "lr": 0.25,
        "betas": (0.5, 0.75),
        "adam_eps": 1e-8,
        "weight_decay": 0.4,
        "weight_decay_policy": "uniform",
        "reference_backend": "mlx",
    }

    optimizer = build_optimizer(model, cfg)
    model.weight.grad = torch.full_like(model.weight, 3.0)
    optimizer.step()

    m = (1.0 - cfg["betas"][0]) * 3.0
    v = (1.0 - cfg["betas"][1]) * 3.0 * 3.0
    adam_updated = 2.0 - cfg["lr"] * m / (math.sqrt(v) + cfg["adam_eps"])
    expected_after_decay = adam_updated * (1.0 - cfg["lr"] * cfg["weight_decay"])
    decay_before_adam = 2.0 * (1.0 - cfg["lr"] * cfg["weight_decay"])
    expected_before_decay = decay_before_adam - cfg["lr"] * m / (math.sqrt(v) + cfg["adam_eps"])

    assert model.weight.item() == pytest.approx(expected_after_decay)
    assert model.weight.item() != pytest.approx(expected_before_decay)


def test_production_like_one_step_optimizer_parity_matches_mlx_reference():
    cfg = _production_like_optimizer_cfg()
    torch_model, mlx_model = _build_pair(cfg)
    state = _deterministic_state(torch_model.state_dict())
    _load_same_state(torch_model, mlx_model, state)
    _, torch_grads = _torch_loss_and_grads(torch_model, cfg)
    _, mlx_grads = _mlx_loss_and_grads(mlx_model, cfg)

    optimizer = build_optimizer(torch_model, cfg)
    for name, param in torch_model.named_parameters():
        param.grad = torch.from_numpy(torch_grads[name].copy())
    optimizer.step()

    mlx_optimizer = mxoptim.AdamW(
        learning_rate=float(cfg["lr"]),
        betas=list(cfg["betas"]),
        eps=float(cfg["adam_eps"]),
        weight_decay=0.0,
    )
    mlx_optimizer.update(mlx_model, mlx_utils.tree_unflatten([(name, mx.array(value)) for name, value in mlx_grads.items()]))
    mlx_backend._apply_decoupled_weight_decay(
        mlx_model,
        mlx_backend._build_weight_decay_lookup(mlx_model, "neo", cfg),
        float(cfg["lr"]),
    )
    mx.eval(mlx_model.parameters(), mlx_optimizer.state)

    torch_after = to_numpy_state_dict(torch_model.state_dict())
    mlx_after = {name: np.asarray(value) for name, value in _mlx_params(mlx_model).items()}
    for name, torch_value in torch_after.items():
        _assert_close(name, torch_value, mlx_after[name], atol=1e-5)


def test_production_like_no_checkpoint_cpu_training_trajectory_stays_close_to_mlx_reference():
    cfg = _production_like_optimizer_cfg()
    torch_model, mlx_model = _build_pair(cfg)
    state = _deterministic_state(torch_model.state_dict())
    _load_same_state(torch_model, mlx_model, state)
    assert cfg["use_checkpoint"] is False

    torch_model.train()
    mlx_model.train()
    torch_optimizer = build_optimizer(torch_model, cfg)
    mlx_optimizer = mxoptim.AdamW(
        learning_rate=float(cfg["lr"]),
        betas=list(cfg["betas"]),
        eps=float(cfg["adam_eps"]),
        weight_decay=0.0,
    )
    mlx_wd_lookup = mlx_backend._build_weight_decay_lookup(mlx_model, "neo", cfg)
    torch_state = torch.from_numpy(_state(cfg))
    mlx_state = mx.array(torch_state.detach().numpy())
    loss_diffs = []
    state_norm_diffs = []

    def mlx_loss_fn(batch_x, batch_y, batch_state):
        logits, new_state = mlx_model(batch_x, batch_state)
        loss = mx.mean(mxnn.losses.cross_entropy(logits.reshape(-1, int(cfg["vocab_size"])), batch_y.reshape(-1)))
        return loss, new_state

    loss_and_grad = mxnn.value_and_grad(mlx_model, mlx_loss_fn)
    for step in range(50):
        x_np, y_np = _tokens(cfg)
        x_np = (x_np + step * 3) % int(cfg["vocab_size"])
        y_np = (y_np + step * 3) % int(cfg["vocab_size"])

        x_torch = torch.from_numpy(x_np).long()
        y_torch = torch.from_numpy(y_np).long()
        torch_optimizer.zero_grad(set_to_none=True)
        torch_logits, torch_state_out = torch_model(x_torch, torch_state)
        torch_loss = F.cross_entropy(torch_logits.reshape(-1, int(cfg["vocab_size"])), y_torch.reshape(-1))
        torch_loss.backward()
        torch_optimizer.step()
        torch_state = torch_state_out.detach()

        x_mlx = mx.array(x_np.astype(np.int32))
        y_mlx = mx.array(y_np.astype(np.int32))
        (mlx_loss, mlx_state_out), mlx_grads = loss_and_grad(x_mlx, y_mlx, mlx_state)
        mlx_optimizer.update(mlx_model, mlx_grads)
        mlx_backend._apply_decoupled_weight_decay(mlx_model, mlx_wd_lookup, float(cfg["lr"]))
        mx.eval(mlx_loss, mlx_state_out, mlx_model.parameters(), mlx_optimizer.state)
        mlx_state = mx.stop_gradient(mlx_state_out)

        loss_diffs.append(abs(float(torch_loss.detach().item()) - float(mlx_loss.item())))
        state_norm_diffs.append(abs(float(torch.linalg.vector_norm(torch_state).item()) - float(mx.linalg.norm(mlx_state).item())))

    torch_after = to_numpy_state_dict(torch_model.state_dict())
    mlx_after = {name: np.asarray(value) for name, value in _mlx_params(mlx_model).items()}
    max_param_diff = max(float(np.max(np.abs(torch_after[name] - mlx_after[name]))) for name in torch_after)
    param_norm_diff = abs(_l2_norm_tree(torch_after) - _l2_norm_tree(mlx_after))
    final_state_diff = float(np.max(np.abs(torch_state.detach().numpy() - np.asarray(mlx_state))))

    assert max(loss_diffs) <= 1e-5
    assert max(state_norm_diffs) <= 1e-5
    assert max_param_diff <= 1e-5
    assert param_norm_diff <= 1e-5
    assert final_state_diff <= 1e-5


def test_checkpoint_roundtrip_loads_both_directions_and_preserves_eval_ce(tmp_path: Path):
    cfg = _tiny_cfg(recurrent_norm="rmsnorm", activation_id="id5", n_layers=2)
    torch_model, mlx_model = _build_pair(cfg)
    state = _deterministic_state(torch_model.state_dict())
    _load_same_state(torch_model, mlx_model, state)

    mlx_path = tmp_path / "reference_mlx.pkl"
    mlx_backend.save_checkpoint_entry(mlx_path, mlx_model, None, None, epoch=1, global_step=2, cfg=cfg)
    torch_from_mlx, _ = _build_pair(cfg)
    ckpt = torch_backend.load_checkpoint_entry(mlx_path, torch_from_mlx, device=torch.device("cpu"))
    for key in ("reference_backend", "rmsnorm_eps", "activation_id", "recurrent_norm", "use_checkpoint"):
        assert ckpt["cfg"][key] == cfg[key]
    torch_model.eval()
    torch_from_mlx.eval()
    ref_logits, _, ref_state, _ = _forward_pair(torch_model, mlx_model, cfg)
    got_logits, _, got_state, _ = _forward_pair(torch_from_mlx, mlx_model, cfg)
    _assert_close("mlx checkpoint to torch logits", got_logits, ref_logits)
    _assert_close("mlx checkpoint to torch state", got_state, ref_state)

    torch_path = tmp_path / "reference_torch.pkl"
    torch_backend.save_checkpoint_entry(torch_path, torch_model, None, None, epoch=1, global_step=2, cfg=cfg)
    _, mlx_from_torch = _build_pair(cfg)
    mlx_backend.load_checkpoint_entry(torch_path, mlx_from_torch)
    _, ref_mlx_logits, _, ref_mlx_state = _forward_pair(torch_model, mlx_model, cfg)
    _, got_mlx_logits, _, got_mlx_state = _forward_pair(torch_model, mlx_from_torch, cfg)
    _assert_close("torch checkpoint to mlx logits", got_mlx_logits, ref_mlx_logits)
    _assert_close("torch checkpoint to mlx state", got_mlx_state, ref_mlx_state)


def test_checkpointed_pytorch_neo_matches_no_checkpoint_gradients_after_closure_binding():
    base_cfg = _tiny_cfg(recurrent_norm="rmsnorm", activation_id="id5", n_layers=2, use_checkpoint=False)
    ckpt_cfg = dict(base_cfg, use_checkpoint=True)
    no_ckpt = torch_backend.build_model(base_cfg, "neo")
    ckpt = torch_backend.build_model(ckpt_cfg, "neo")
    state = _deterministic_state(no_ckpt.state_dict())
    no_ckpt.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in state.items()})
    ckpt.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in state.items()})

    _, no_ckpt_grads = _torch_loss_and_grads(no_ckpt, base_cfg)
    _, ckpt_grads = _torch_loss_and_grads(ckpt, ckpt_cfg)

    for name, expected in no_ckpt_grads.items():
        _assert_close(name, ckpt_grads[name], expected, atol=1e-5)
