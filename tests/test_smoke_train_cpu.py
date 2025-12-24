import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import LSTMLM
from src.train import train_model


def test_smoke_train_cpu(tmp_path):
    cfg = {
        "vocab_size": 32,
        "d_model": 16,
        "d_embed": 8,
        "n_layers": 2,
        "dropout": 0.0,
        "tie_embeddings": True,
        "block_size": 8,
        "batch_size": 4,
        "epochs": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "cosine": False,
        "save_dir": str(tmp_path),
        "run_tag": "smoke",
    }
    model = LSTMLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        d_embed=cfg["d_embed"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        tie_embeddings=cfg["tie_embeddings"],
    )

    train_ids = torch.randint(0, cfg["vocab_size"], (256,))
    val_ids = torch.randint(0, cfg["vocab_size"], (128,))

    metrics = train_model(
        model,
        cfg,
        train_ids,
        val_ids,
        test_ids=None,
        device=torch.device("cpu"),
    )

    assert "val_ppl" in metrics
    assert (tmp_path / "best_smoke.pt").exists()
    assert (tmp_path / "last_smoke.pt").exists()
