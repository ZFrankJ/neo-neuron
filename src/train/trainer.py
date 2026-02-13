"""Unified training loop for Neo models."""

import gc
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..data.batching import get_batch
from .checkpointing import load_checkpoint, save_checkpoint
from .eval import eval_perplexity, evaluate_metrics, init_state_for_model
from .logging import log_line, maybe_report_memory, maybe_clear_memory
from .optim import build_optimizer
from .schedulers import build_scheduler


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _count_params(module: Optional[torch.nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _param_breakdown(model: torch.nn.Module) -> Tuple[int, Dict[str, int]]:
    total = _count_params(model)
    if hasattr(model, "lstm"):
        emb = model.emb.weight.numel()
        recurrent = (
            _count_params(model.lstm)
            + _count_params(getattr(model, "pre_norm", None))
            + _count_params(getattr(model, "stack_norm", None))
        )
        proj = _count_params(model.in_proj) + _count_params(model.out_proj)
        head = 0
        if getattr(model, "output_bias", None) is not None:
            head += model.output_bias.numel()
        if getattr(model, "head", None) is not None:
            head += _count_params(model.head)
        return total, {"embeddings": emb, "recurrent": recurrent, "proj_head": proj + head}
    if hasattr(model, "recurrent"):
        emb = model.emb.weight.numel()
        recurrent = _count_params(model.recurrent)
        proj = _count_params(model.in_proj) + _count_params(model.out_proj)
        head = 0
        if getattr(model, "output_bias", None) is not None:
            head += model.output_bias.numel()
        if getattr(model, "head", None) is not None:
            head += _count_params(model.head)
        return total, {"embeddings": emb, "recurrent": recurrent, "proj_head": proj + head}
    if hasattr(model, "blocks"):
        emb = model.emb.weight.numel() + model.pos_emb.weight.numel()
        core = _count_params(model.blocks) + _count_params(model.ln_f)
        head = 0
        if getattr(model, "output_bias", None) is not None:
            head += model.output_bias.numel()
        if getattr(model, "head", None) is not None:
            head += _count_params(model.head)
        return total, {"embeddings": emb, "transformer": core, "head": head}
    return total, {}


def _log_model_info(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    total, breakdown = _param_breakdown(model)
    model_tag = _cfg_get(cfg, "model_name", None) if cfg is not None else None
    log_line("== Model Info ==")
    if model_tag:
        log_line(f"Model tag: {model_tag}")
    log_line(f"Model: {model.__class__.__name__}")
    log_line(f"Total params: {total / 1e6:.2f}M")
    if breakdown:
        parts = [f"{key}={val / 1e6:.2f}M" for key, val in breakdown.items()]
        log_line("Breakdown: " + " | ".join(parts))


def _write_progress(run_dir: Optional[str], payload: Dict[str, float]) -> None:
    if not run_dir:
        return
    os.makedirs(run_dir, exist_ok=True)
    progress_path = os.path.join(run_dir, "progress.json")
    history_path = os.path.join(run_dir, "history.jsonl")
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _is_recurrent_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "recurrent") or hasattr(model, "lstm")


def _detach_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()


def _state_batch_size(state) -> Optional[int]:
    if state is None:
        return None
    if isinstance(state, tuple):
        if not state:
            return None
        first = state[0]
        if hasattr(first, "size") and first.dim() >= 2:
            return int(first.size(1))
        return None
    if hasattr(state, "size") and state.dim() >= 2:
        return int(state.size(1))
    return None


def _streaming_batches(ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    n_tokens = ids.size(0)
    step = block_size * batch_size
    for start in range(0, n_tokens - (block_size + 1), step):
        cur_B = min(batch_size, (n_tokens - (start + block_size + 1)) // block_size)
        if cur_B <= 0:
            break
        x = torch.stack(
            [ids[start + i * block_size : start + i * block_size + block_size] for i in range(cur_B)]
        ).to(device)
        y = torch.stack(
            [ids[start + i * block_size + 1 : start + i * block_size + block_size + 1] for i in range(cur_B)]
        ).to(device)
        yield x.t().contiguous(), y.t().contiguous(), cur_B


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
    log_line(f"Using device: {device}")

    _log_model_info(model, cfg)

    if _cfg_get(cfg, "use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = build_optimizer(model, cfg)
    batch_size = int(_cfg_get(cfg, "batch_size", 1))
    block_size = int(_cfg_get(cfg, "block_size", 1))
    steps_per_epoch = max(1, train_ids.size(0) // (batch_size * block_size))
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
        # Avoid retaining a full checkpoint dict (model + optimizer states) in memory.
        del ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    epochs = int(_cfg_get(cfg, "epochs", 1))
    grad_clip = float(_cfg_get(cfg, "grad_clip", 1.0))
    mem_interval = _cfg_get(cfg, "mem_report_interval", None)
    mem_clear_interval = _cfg_get(cfg, "mem_clear_interval", None)

    train_regime = str(_cfg_get(cfg, "train_regime", "random")).lower()
    stream_state = bool(_cfg_get(cfg, "stream_state", False))
    if train_regime not in ("random", "streaming"):
        raise ValueError(f"Unsupported train_regime '{train_regime}'.")

    run_dir = _cfg_get(cfg, "run_dir", None)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        if train_regime == "random":
            step_iter = range(steps_per_epoch)
            state = None
        else:
            step_iter = _streaming_batches(train_ids, block_size, batch_size, device)
            state = None

        for step in step_iter:
            if train_regime == "random":
                x, y = get_batch(train_ids, batch_size, block_size, device)
                cur_B = x.size(1)
            else:
                x, y, cur_B = step

            tbptt_len = int(_cfg_get(cfg, "tbptt_len", 0))
            optimizer.zero_grad(set_to_none=True)

            if _is_recurrent_model(model):
                if not stream_state or state is None or (_state_batch_size(state) != cur_B):
                    state = init_state_for_model(model, cur_B, device)
            else:
                state = None

            if tbptt_len > 0 and tbptt_len < x.size(0) and _is_recurrent_model(model):
                batch_loss = torch.zeros((), device=device)
                chunks = 0
                for start in range(0, x.size(0), tbptt_len):
                    end = min(x.size(0), start + tbptt_len)
                    logits, state = model(x[start:end], state)
                    loss = F.cross_entropy(
                        logits.reshape(-1, int(_cfg_get(cfg, "vocab_size", 0))),
                        y[start:end].reshape(-1),
                    )
                    loss_total = loss
                    loss_total.backward()
                    batch_loss += loss_total.detach()
                    chunks += 1
                    state = _detach_state(state)
                epoch_loss += batch_loss / max(1, chunks)
                if not stream_state:
                    state = None
            else:
                logits, state_out = model(x, state)
                loss = F.cross_entropy(
                    logits.reshape(-1, int(_cfg_get(cfg, "vocab_size", 0))),
                    y.reshape(-1),
                )
                loss_total = loss
                loss_total.backward()
                epoch_loss += loss_total.detach()
                if _is_recurrent_model(model) and stream_state:
                    state = _detach_state(state_out)
                else:
                    state = None

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            maybe_report_memory(global_step, mem_interval)
            maybe_clear_memory(global_step, mem_clear_interval)

        val_ppl = eval_perplexity(model, val_ids, cfg, device)
        train_ce = float(epoch_loss / steps_per_epoch)
        log_line(
            f"Epoch {epoch:02d}/{epochs} | Train CE: {train_ce:.4f} | Val PPL: {val_ppl:.2f}"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, global_step, cfg, best_val)
        if val_ppl < best_val:
            best_val = val_ppl
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, global_step, cfg, best_val)

        if _cfg_get(cfg, "save_each_epoch", False):
            epoch_path = os.path.join(save_dir, f"epoch_{epoch:02d}_{run_tag}.pt")
            save_checkpoint(epoch_path, model, optimizer, scheduler, epoch, global_step, cfg, best_val)

        _write_progress(
            run_dir,
            {
                "epoch": epoch,
                "train_ce": train_ce,
                "val_ppl": val_ppl,
                "best_val_ppl": best_val,
                "global_step": global_step,
            },
        )

    metrics = {"val_ppl": best_val}
    if test_ids is not None:
        eval_metrics = evaluate_metrics(model, test_ids, cfg, device)
        metrics["test_ppl"] = eval_metrics["ppl"]
        if eval_metrics.get("gflops_per_token") is not None:
            metrics["gflops_per_token"] = eval_metrics["gflops_per_token"]
        if eval_metrics.get("act_sparsity") is not None:
            metrics["act_sparsity"] = eval_metrics["act_sparsity"]
        log_line(f"Test PPL: {eval_metrics['ppl']:.2f}")
        if eval_metrics.get("gflops_per_token") is not None:
            log_line(f"Measured GFLOPs/token (THOP): {eval_metrics['gflops_per_token']:.3f}")
        else:
            log_line("Measured GFLOPs/token (THOP): unavailable")
        if eval_metrics.get("act_sparsity") is not None:
            log_line(f"Activation sparsity: {eval_metrics['act_sparsity']:.4f}")

    return metrics
