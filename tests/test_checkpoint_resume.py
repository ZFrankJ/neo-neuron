import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import LSTMLM
from src.train import load_checkpoint, save_checkpoint
from src.train.optim import build_optimizer
from src.train.schedulers import build_scheduler


def test_checkpoint_roundtrip(tmp_path):
    cfg = {
        "vocab_size": 32,
        "d_model": 16,
        "d_embed": 8,
        "n_layers": 2,
        "dropout": 0.0,
        "tie_embeddings": True,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "cosine": False,
        "epochs": 1,
        "warmup_epochs": 0,
        "min_lr": 0.0,
    }
    model = LSTMLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        d_embed=cfg["d_embed"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        tie_embeddings=cfg["tie_embeddings"],
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=1)

    ckpt_path = tmp_path / "ckpt.pt"
    original = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(
        str(ckpt_path),
        model,
        optimizer,
        scheduler,
        epoch=2,
        global_step=7,
        cfg=cfg,
        best_val=3.14,
    )

    for param in model.parameters():
        param.data.add_(1.0)

    ckpt = load_checkpoint(str(ckpt_path), model, optimizer, scheduler)
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, original[name])

    assert ckpt["epoch"] == 2
    assert ckpt["global_step"] == 7
    assert ckpt["best_val"] == 3.14
