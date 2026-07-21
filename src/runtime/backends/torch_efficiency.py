"""PyTorch execution, synchronization, and telemetry benchmark adapter."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from ...train.optim import build_optimizer
from ..efficiency import BenchmarkObservation, hardware_identifier


class TorchEfficiencyAdapter:
    backend = "torch"
    framework_version = str(torch.__version__)

    def __init__(self, model: torch.nn.Module, cfg: Mapping[str, Any], device: torch.device):
        self.model = model
        self.cfg = dict(cfg)
        self.torch_device = torch.device(device)
        self.device = str(self.torch_device)
        if self.torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
        if self.torch_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS benchmark requested but torch.backends.mps.is_available() is false")
        if self.torch_device.type not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported Torch benchmark device '{self.torch_device.type}'")
        parameter = next((value for value in model.parameters() if value.requires_grad), None)
        self.dtype = str(parameter.dtype).replace("torch.", "") if parameter is not None else "unknown"
        accelerator = None
        if self.torch_device.type == "cuda":
            accelerator = torch.cuda.get_device_name(self.torch_device)
        self.hardware_identifier = hardware_identifier(accelerator)
        self.synchronization_policy = {
            "cpu": "torch eager CPU completion",
            "mps": "torch.mps.synchronize()",
            "cuda": f"torch.cuda.synchronize({self.device})",
        }[self.torch_device.type]
        self.telemetry_capabilities = {
            "process_rss": True,
            "backend_active_memory": self.torch_device.type in {"mps", "cuda"},
            "backend_peak_memory": self.torch_device.type == "cuda",
        }
        self.tokens: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.optimizer = None
        self.workload = None

    def parameter_entries(self):
        return [
            (name, int(parameter.numel()), bool(parameter.requires_grad))
            for name, parameter in self.model.named_parameters()
        ]

    def prepare(self, tokens: np.ndarray, targets: np.ndarray) -> None:
        self.tokens = torch.from_numpy(tokens.astype(np.int64, copy=False)).to(self.torch_device)
        self.targets = torch.from_numpy(targets.astype(np.int64, copy=False)).to(self.torch_device)

    def configure(self, workload: str) -> None:
        self.workload = workload
        if workload == "train_step":
            self.model.train()
            self.optimizer = build_optimizer(self.model, self.cfg)
        else:
            self.model.eval()

    def reset_peak_memory(self) -> None:
        if self.torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.torch_device)

    def synchronize(self) -> None:
        if self.torch_device.type == "mps":
            torch.mps.synchronize()
        elif self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)

    def _forward_sequence(self):
        assert self.tokens is not None
        return self.model(self.tokens, None)[0]

    def _forward_streaming(self):
        assert self.tokens is not None
        state = None
        outputs = []
        if hasattr(self.model, "init_state"):
            for index in range(self.tokens.shape[0]):
                logits, state = self.model(self.tokens[index : index + 1], state)
                outputs.append(logits)
        else:
            for index in range(self.tokens.shape[0]):
                logits, _ = self.model(self.tokens[: index + 1], None)
                outputs.append(logits[-1:])
        return torch.cat(outputs, dim=0)

    def run(self, workload: str):
        assert self.tokens is not None and self.targets is not None
        if workload != self.workload:
            raise ValueError(f"Adapter configured for {self.workload!r}, got {workload!r}")
        if workload == "train_step":
            assert self.optimizer is not None
            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward_sequence()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.targets.reshape(-1))
            loss.backward()
            grad_clip = float(self.cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            return {"logits": logits, "scalar": loss}
        with torch.no_grad():
            logits = self._forward_streaming() if workload == "streaming_decode" else self._forward_sequence()
        return {"logits": logits, "scalar": logits.float().mean()}

    def finalize_observation(self, pending) -> BenchmarkObservation:
        logits = pending["logits"]
        scalar = pending["scalar"]
        finite = bool(torch.isfinite(logits).all().item()) and bool(torch.isfinite(scalar).item())
        return BenchmarkObservation(
            output_shape=tuple(int(value) for value in logits.shape),
            finite=finite,
            scalar=float(scalar.detach().cpu().item()),
        )

    def memory_snapshot(self):
        active = None
        peak = None
        if self.torch_device.type == "mps":
            active = int(torch.mps.current_allocated_memory())
        elif self.torch_device.type == "cuda":
            active = int(torch.cuda.memory_allocated(self.torch_device))
            peak = int(torch.cuda.max_memory_allocated(self.torch_device))
        return {
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
            "backend_active_bytes": active,
            "backend_peak_bytes": peak,
        }
