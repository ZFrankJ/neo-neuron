"""Unified training loop for Neo models."""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..data.batching import get_batch
from .checkpointing import load_checkpoint, save_checkpoint
from .eval import eval_perplexity, init_state_for_model
from .logging import log_line, maybe_report_memory
from .optim import build_optimizer
from .schedulers import build_scheduler


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def train_model(
    model: torch.nn.Module,
    cfg: Any,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    test_ids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)

    if _cfg_get(cfg, "use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, train_ids.size(0) // (int(_cfg_get(cfg, "batch_size", 1)) * int(_cfg_get(cfg, "block_size", 1))))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    save_dir = _cfg_get(cfg, "save_dir", "checkpoints")
    run_tag = _cfg_get(cfg, "run_tag", "run")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"best_{run_tag}.pt")
    last_path = os.path.join(save_dir, f"last_{run_tag}.pt")

    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    resume_path = _cfg_get(cfg, "resume_path", "")
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        log_line(f"Resumed from {resume_path} (epoch {start_epoch - 1}, global_step {global_step})")

    epochs = int(_cfg_get(cfg, "epochs", 1))
    grad_clip = float(_cfg_get(cfg, "grad_clip", 1.0))
    mem_interval = _cfg_get(cfg, "mem_report_interval", None)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            x, y = get_batch(
                train_ids,
                int(_cfg_get(cfg, "batch_size", 1)),
                int(_cfg_get(cfg, "block_size", 1)),
                device,
            )
            state = init_state_for_model(model, x.size(1), device)
            logits, _ = model(x, state)
            loss = F.cross_entropy(logits.reshape(-1, int(_cfg_get(cfg, "vocab_size", 0))), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            maybe_report_memory(global_step, mem_interval)

        val_ppl = eval_perplexity(model, val_ids, cfg, device)
        log_line(
            f"Epoch {epoch:02d}/{epochs} | Train CE: {epoch_loss/steps_per_epoch:.4f} | Val PPL: {val_ppl:.2f}"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, global_step, cfg, best_val)
        if val_ppl < best_val:
            best_val = val_ppl
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, global_step, cfg, best_val)

    metrics = {"val_ppl": best_val}
    if test_ids is not None:
        test_ppl = eval_perplexity(model, test_ids, cfg, device)
        metrics["test_ppl"] = test_ppl
        log_line(f"Test PPL: {test_ppl:.2f}")

    return metrics
