import math
from pathlib import Path

import pytest
import torch
import yaml

from src.models import TransformerLM
from src.runtime.backends import torch_backend


def _tiny_model(**overrides):
    kwargs = dict(
        vocab_size=97,
        d_model=64,
        n_layers=4,
        n_heads=4,
        ff_mult=4,
        dropout=0.0,
        tie_embeddings=True,
        max_seq_len=32,
        transformer_variant="gpt2",
    )
    kwargs.update(overrides)
    return TransformerLM(**kwargs)


def test_gpt2_attention_uses_optimized_causal_primitive(monkeypatch):
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    model = _tiny_model(n_layers=2).eval()
    logits, state = model(torch.randint(0, 97, (7, 3)), None)

    assert logits.shape == (7, 3, 97)
    assert state is None
    assert len(calls) == 2
    assert all(call["is_causal"] is True for call in calls)
    assert all(call["dropout_p"] == 0.0 for call in calls)


def test_gpt2_initialization_scales_residual_projections():
    torch.manual_seed(7)
    model = _tiny_model()
    base_std = model.blocks[0].attn.qkv.weight.std().item()
    attn_residual_std = model.blocks[0].attn.out_proj.weight.std().item()
    mlp_residual_std = model.blocks[0].mlp[3].weight.std().item()
    expected_residual_std = 0.02 / math.sqrt(2 * model.n_layers)

    assert base_std == pytest.approx(0.02, rel=0.15)
    assert attn_residual_std == pytest.approx(expected_residual_std, rel=0.15)
    assert mlp_residual_std == pytest.approx(expected_residual_std, rel=0.15)
    assert model.blocks[0].attn.qkv.bias is None


def test_gpt2_causality_blocks_future_token_changes():
    torch.manual_seed(11)
    model = _tiny_model(n_layers=2).eval()
    prefix = torch.tensor([[1], [2], [3], [4]])
    changed_future = prefix.clone()
    changed_future[-1] = 9

    first, _ = model(prefix, None)
    second, _ = model(changed_future, None)

    torch.testing.assert_close(first[:-1], second[:-1], rtol=0.0, atol=0.0)


def test_transformer_config_explicitly_labels_gpt2_control():
    cfg = yaml.safe_load(Path("configs/wt103/transformer_30m.yaml").read_text())
    assert cfg["transformer_variant"] == "gpt2"
    assert "gpt2" in cfg["run_tag"]

    model = torch_backend.build_model(cfg, "transformer")
    assert model.transformer_variant == "gpt2"


def test_unknown_transformer_variant_is_rejected():
    with pytest.raises(ValueError, match="transformer_variant"):
        _tiny_model(transformer_variant="modernish")


def test_gpt2_mlx_and_torch_checkpoint_forward_parity(tmp_path):
    mlx_backend = pytest.importorskip("src.runtime.backends.mlx_backend")
    import mlx.core as mx

    cfg = {
        "backend": "torch",
        "model_name": "transformer",
        "transformer_variant": "gpt2",
        "vocab_size": 31,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "ff_mult": 2,
        "dropout": 0.0,
        "tie_embeddings": True,
        "block_size": 16,
        "recurrent_norm": "none",
    }
    torch.manual_seed(13)
    torch_model = torch_backend.build_model(cfg, "transformer").eval()
    checkpoint = tmp_path / "gpt2.pt"
    torch_backend.save_checkpoint_entry(
        checkpoint, torch_model, None, None, epoch=0, global_step=0, cfg=cfg
    )

    mlx_model = mlx_backend.build_model(cfg, "transformer")
    mlx_backend.load_checkpoint_entry(checkpoint, mlx_model, cfg=cfg)
    mlx_model.eval()
    ids = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    torch_logits, _ = torch_model(ids, None)
    mlx_logits, _ = mlx_model(mx.array(ids.numpy()), None)
    mx.eval(mlx_logits)

    torch.testing.assert_close(
        torch_logits.detach(),
        torch.from_numpy(__import__("numpy").array(mlx_logits)),
        rtol=2e-4,
        atol=2e-5,
    )
