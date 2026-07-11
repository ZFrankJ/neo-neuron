import pytest

from src.runtime.backends import torch_backend
from src.train.optim import build_optimizer


def _cfg(**overrides):
    cfg = {
        "vocab_size": 32,
        "d_model": 8,
        "d_embed": 4,
        "n_layers": 2,
        "n_heads": 2,
        "ff_mult": 2,
        "block_size": 8,
        "dropout": 0.0,
        "tie_embeddings": True,
        "recurrent_norm": "rmsnorm",
        "recurrent_norm_place": "all",
        "activation_id": "id5",
        "use_checkpoint": False,
        "reference_backend": "mlx",
        "weight_decay_policy": "table",
        "embed_weight_decay": 1e-4,
        "proj_weight_decay": 1e-3,
        "recurrent_weight_decay": 2e-4,
        "transformer_weight_decay": 1e-2,
        "lr": 3e-4,
    }
    cfg.update(overrides)
    return cfg


def _weight_decay_by_name(model, optimizer):
    decay_by_param = {
        id(param): float(group["weight_decay"])
        for group in optimizer.param_groups
        for param in group["params"]
    }
    return {name: decay_by_param[id(param)] for name, param in model.named_parameters()}


def test_mlx_reference_optimizer_groups_torch_lstm_parameters_by_role():
    cfg = _cfg()
    model = torch_backend.build_model(cfg, "lstm")

    decay = _weight_decay_by_name(model, build_optimizer(model, cfg))

    assert decay["emb.weight"] == pytest.approx(cfg["embed_weight_decay"])
    assert decay["in_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert decay["out_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert decay["lstm.weight_ih_l0"] == pytest.approx(cfg["recurrent_weight_decay"])
    assert decay["lstm.weight_hh_l1"] == pytest.approx(cfg["recurrent_weight_decay"])
    assert decay["lstm.bias_ih_l0"] == pytest.approx(0.0)
    assert decay["lstm.pre_norms.0.weight"] == pytest.approx(0.0)
    assert decay["lstm.stack_norm.weight"] == pytest.approx(0.0)
    assert decay["output_bias"] == pytest.approx(0.0)


def test_mlx_reference_optimizer_keeps_neo_grouping_unchanged():
    cfg = _cfg()
    model = torch_backend.build_model(cfg, "neo")

    decay = _weight_decay_by_name(model, build_optimizer(model, cfg))

    assert decay["emb.weight"] == pytest.approx(cfg["embed_weight_decay"])
    assert decay["in_proj.weight"] == pytest.approx(cfg["proj_weight_decay"])
    assert decay["recurrent.layers.0.fg_linear.weight"] == pytest.approx(
        cfg["recurrent_weight_decay"]
    )
    assert decay["recurrent.layers.0.fg_linear.bias"] == pytest.approx(0.0)


def test_mlx_reference_optimizer_keeps_transformer_grouping_unchanged():
    cfg = _cfg(transformer_variant="gpt2")
    model = torch_backend.build_model(cfg, "transformer")

    decay = _weight_decay_by_name(model, build_optimizer(model, cfg))

    assert decay["emb.weight"] == pytest.approx(cfg["embed_weight_decay"])
    assert decay["pos_emb.weight"] == pytest.approx(cfg["embed_weight_decay"])
    assert decay["blocks.0.attn.qkv.weight"] == pytest.approx(
        cfg["transformer_weight_decay"]
    )
    assert decay["blocks.0.ln1.weight"] == pytest.approx(0.0)
