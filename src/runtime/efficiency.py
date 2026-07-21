"""Backend-neutral wall-clock and memory benchmark contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .backend import get_backend
from .checkpoint_compat import infer_checkpoint_backend, load_checkpoint_payload


SCHEMA_VERSION = "neo_efficiency_benchmark_v1"
SUPPORTED_WORKLOADS = {"train_step", "sequence_eval", "streaming_decode"}
SUPPORTED_TIMING_SCOPES = {"model_only", "end_to_end_loop"}
SUPPORTED_WEIGHT_PROVENANCE = {
    "mapped_same_checkpoint",
    "backend_native_checkpoint",
}
MIN_WARMUP_ITERATIONS = 20
MIN_MEASURED_ITERATIONS = 100
WORKLOAD_SEMANTICS = {
    "train_step": {
        "input_policy": "deterministic_preallocated",
        "optimizer_state_policy": "fresh_then_warmed",
        "scheduler_included": False,
        "state_policy": "reset_each_iteration",
        "tbptt_policy": "full_sequence",
    },
    "sequence_eval": {
        "input_policy": "deterministic_preallocated",
        "state_policy": "reset_each_iteration",
    },
    "streaming_decode": {
        "input_policy": "deterministic_preallocated",
        "state_policy": "reset_for_each_sequence",
    },
}


@dataclass(frozen=True)
class BenchmarkObservation:
    output_shape: Sequence[int]
    finite: bool
    scalar: float


@dataclass(frozen=True)
class BenchmarkSpec:
    workload: str
    timing_scope: str
    batch_size: int
    sequence_length: int
    warmup_iterations: int
    measured_iterations: int
    repetition_id: str
    seed: int
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.workload not in SUPPORTED_WORKLOADS:
            raise ValueError(
                f"Unsupported workload '{self.workload}'. Expected one of: "
                f"{sorted(SUPPORTED_WORKLOADS)}"
            )
        if self.timing_scope not in SUPPORTED_TIMING_SCOPES:
            raise ValueError(
                f"Unsupported timing_scope '{self.timing_scope}'. Expected one of: "
                f"{sorted(SUPPORTED_TIMING_SCOPES)}"
            )
        expected_scope = (
            "end_to_end_loop" if self.workload == "train_step" else "model_only"
        )
        if self.timing_scope != expected_scope:
            raise ValueError(
                f"workload '{self.workload}' requires timing_scope '{expected_scope}', "
                f"got '{self.timing_scope}'"
            )
        for name in (
            "batch_size",
            "sequence_length",
            "measured_iterations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.warmup_iterations) < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if not str(self.repetition_id).strip():
            raise ValueError("repetition_id must be non-empty")
        if not self.dry_run and self.warmup_iterations < MIN_WARMUP_ITERATIONS:
            raise ValueError(
                f"Formal records require at least {MIN_WARMUP_ITERATIONS} warm-up iterations"
            )
        if not self.dry_run and self.measured_iterations < MIN_MEASURED_ITERATIONS:
            raise ValueError(
                f"Formal records require at least {MIN_MEASURED_ITERATIONS} measured iterations"
            )


class EfficiencyAdapter(Protocol):
    backend: str
    device: str
    dtype: str
    framework_version: str
    hardware_identifier: str
    synchronization_policy: str
    telemetry_capabilities: Mapping[str, bool]

    def prepare(self, tokens: np.ndarray, targets: np.ndarray) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def synchronize(self) -> None: ...

    def run(self, workload: str) -> Any: ...

    def memory_snapshot(self) -> Mapping[str, int | None]: ...


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(samples_ns: Sequence[int], tokens_per_iteration: int) -> Dict[str, float]:
    median_ns = _percentile(samples_ns, 0.5)
    return {
        "min_ns": float(min(samples_ns)),
        "median_ns": median_ns,
        "q1_ns": _percentile(samples_ns, 0.25),
        "q3_ns": _percentile(samples_ns, 0.75),
        "p10_ns": _percentile(samples_ns, 0.10),
        "p90_ns": _percentile(samples_ns, 0.90),
        "max_ns": float(max(samples_ns)),
        "tokens_per_second": float(tokens_per_iteration * 1_000_000_000 / median_ns),
        "milliseconds_per_token": float(median_ns / 1_000_000 / tokens_per_iteration),
        "milliseconds_per_step": float(median_ns / 1_000_000),
    }


def _finalize_observation(adapter: EfficiencyAdapter, pending: Any) -> BenchmarkObservation:
    finalize = getattr(adapter, "finalize_observation", None)
    observation = finalize(pending) if callable(finalize) else pending
    if not isinstance(observation, BenchmarkObservation):
        raise TypeError("Benchmark adapter must return BenchmarkObservation")
    return observation


def _deterministic_inputs(spec: BenchmarkSpec, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    size = spec.sequence_length * spec.batch_size
    values = (
        np.arange(size, dtype=np.int64).reshape(spec.sequence_length, spec.batch_size)
        * 17
        + int(spec.seed)
    ) % vocab_size
    tokens = np.ascontiguousarray(values.astype(np.int32))
    targets = np.ascontiguousarray(((values + 1) % vocab_size).astype(np.int32))
    return tokens, targets


def hardware_identifier(accelerator: str | None = None) -> str:
    """Return a stable, host-independent platform and processor model label."""

    system = platform.system() or "unknown"
    machine = platform.machine() or "unknown"
    details: list[str] = []
    if system == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.run(
                    ["sysctl", "-n", key],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                value = ""
            if value and value not in details:
                details.append(value)
    elif system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            cpuinfo = ""
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    details.append(value)
                break
    if not details:
        processor = platform.processor().strip()
        if processor and processor != machine:
            details.append(processor)
    if accelerator and accelerator.strip() and accelerator.strip() not in details:
        details.append(accelerator.strip())
    return " | ".join([system, machine, *details])


def run_benchmark(
    adapter: EfficiencyAdapter,
    spec: BenchmarkSpec,
    metadata: Mapping[str, Any],
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> Dict[str, Any]:
    identity = dict(metadata.get("identity", {}))
    model = dict(metadata.get("model", {}))
    vocab_size = int(identity.get("config_snapshot", {}).get("vocab_size", 0))
    tokens, targets = _deterministic_inputs(spec, vocab_size)
    configure = getattr(adapter, "configure", None)
    if callable(configure):
        configure(spec.workload)
    adapter.prepare(tokens, targets)

    last_observation: BenchmarkObservation | None = None
    for _ in range(spec.warmup_iterations):
        pending = adapter.run(spec.workload)
        adapter.synchronize()
        last_observation = _finalize_observation(adapter, pending)
        if not last_observation.finite:
            raise ValueError("Benchmark produced non-finite output during warm-up")

    adapter.reset_peak_memory()
    samples_ns = []
    memory: Dict[str, int | None] = {}
    peak_bytes: int | None = None
    for index in range(spec.measured_iterations):
        adapter.synchronize()
        started = int(clock_ns())
        pending = adapter.run(spec.workload)
        adapter.synchronize()
        finished = int(clock_ns())
        elapsed = finished - started
        if elapsed <= 0:
            raise ValueError(f"Measured duration must be positive, got {elapsed} ns")
        samples_ns.append(elapsed)
        snapshot = dict(adapter.memory_snapshot())
        memory.update(snapshot)
        measured_peak = snapshot.get("backend_peak_bytes")
        if measured_peak is not None:
            peak_bytes = max(peak_bytes or 0, int(measured_peak))
            memory["backend_peak_bytes"] = peak_bytes
        last_observation = _finalize_observation(adapter, pending)
        if not last_observation.finite:
            raise ValueError("Benchmark produced non-finite output")
        if index + 1 < spec.measured_iterations:
            adapter.reset_peak_memory()

    assert last_observation is not None
    tokens_per_iteration = spec.batch_size * spec.sequence_length
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "model": model,
        "runtime": {
            "backend": str(adapter.backend),
            "device": str(adapter.device),
            "requested_device": str(getattr(adapter, "requested_device", adapter.device)),
            "dtype": str(adapter.dtype),
            "framework_version": str(adapter.framework_version),
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "hardware_identifier": str(adapter.hardware_identifier),
            "synchronization_policy": str(adapter.synchronization_policy),
        },
        "workload": {
            "name": spec.workload,
            "timing_scope": spec.timing_scope,
            "batch_size": spec.batch_size,
            "sequence_length": spec.sequence_length,
            "logical_shape": [spec.sequence_length, spec.batch_size],
            "warmup_iterations": spec.warmup_iterations,
            "measured_iterations": spec.measured_iterations,
            "repetition_id": spec.repetition_id,
            "seed": spec.seed,
            "tokens_per_iteration": tokens_per_iteration,
            "data_handling_included": False,
            "semantics": dict(WORKLOAD_SEMANTICS[spec.workload]),
            "dry_run": spec.dry_run,
        },
        "telemetry_capabilities": dict(adapter.telemetry_capabilities),
        "memory": memory,
        "raw_samples_ns": samples_ns,
        "summary": _summary(samples_ns, tokens_per_iteration),
        "output": {
            "shape": [int(value) for value in last_observation.output_shape],
            "finite": bool(last_observation.finite),
            "last_scalar": float(last_observation.scalar),
        },
    }
    validate_benchmark_record(record)
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["record_id"] = hashlib.sha256(canonical).hexdigest()
    return record


def _walk_finite(value: Any, path: str = "record") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite numeric value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_finite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_finite(child, f"{path}[{index}]")
        return
    raise TypeError(f"Unsupported record value at {path}: {type(value).__name__}")


def validate_benchmark_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record.get('schema_version')!r}")
    required_mapping_fields = {
        "identity": {
            "git_commit",
            "config_path",
            "config_sha256",
            "config_snapshot",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_backend",
            "checkpoint_metadata",
        },
        "model": {
            "name",
            "profile_label",
            "weight_provenance",
            "trainable_parameters",
            "parameter_breakdown",
            "use_checkpoint",
        },
        "runtime": {
            "backend",
            "device",
            "requested_device",
            "dtype",
            "framework_version",
            "python_version",
            "os",
            "hardware_identifier",
            "synchronization_policy",
        },
        "summary": {
            "min_ns",
            "median_ns",
            "q1_ns",
            "q3_ns",
            "p10_ns",
            "p90_ns",
            "max_ns",
            "tokens_per_second",
            "milliseconds_per_token",
            "milliseconds_per_step",
        },
    }
    for section, required in required_mapping_fields.items():
        value = record.get(section)
        if not isinstance(value, Mapping):
            raise ValueError(f"Record is missing {section} metadata")
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Record {section} metadata is missing fields: {missing}")
    workload = record.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("Record is missing workload metadata")
    BenchmarkSpec(
        workload=str(workload.get("name", "")),
        timing_scope=str(workload.get("timing_scope", "")),
        batch_size=int(workload.get("batch_size", 0)),
        sequence_length=int(workload.get("sequence_length", 0)),
        warmup_iterations=int(workload.get("warmup_iterations", -1)),
        measured_iterations=int(workload.get("measured_iterations", 0)),
        repetition_id=str(workload.get("repetition_id", "")),
        seed=int(workload.get("seed", 0)),
        dry_run=bool(workload.get("dry_run", False)),
    )
    samples = record.get("raw_samples_ns")
    if not isinstance(samples, list) or not samples:
        raise ValueError("raw_samples_ns must be a non-empty list")
    if len(samples) != int(workload.get("measured_iterations", -1)):
        raise ValueError("raw_samples_ns length does not match measured_iterations")
    if any(not isinstance(value, int) or value <= 0 for value in samples):
        raise ValueError("raw_samples_ns must contain positive integer durations")
    if workload.get("data_handling_included") is not False:
        raise ValueError("Benchmark records must exclude data handling from measured regions")
    expected_semantics = WORKLOAD_SEMANTICS[str(workload["name"])]
    if workload.get("semantics") != expected_semantics:
        raise ValueError("Benchmark workload semantics do not match the workload contract")
    expected_tokens = int(workload["batch_size"]) * int(workload["sequence_length"])
    if workload.get("tokens_per_iteration") != expected_tokens:
        raise ValueError("tokens_per_iteration does not match the logical workload shape")
    expected_summary = _summary(samples, expected_tokens)
    summary = record["summary"]
    for key, expected in expected_summary.items():
        if not math.isclose(float(summary[key]), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Summary field '{key}' does not match raw samples")
    model = record["model"]
    if model.get("weight_provenance") not in SUPPORTED_WEIGHT_PROVENANCE:
        raise ValueError("Record has unsupported weight_provenance")
    if model.get("use_checkpoint") is not False:
        raise ValueError("Efficiency benchmark records require use_checkpoint=false")
    output = record.get("output")
    if not isinstance(output, Mapping) or output.get("finite") is not True:
        raise ValueError("Benchmark output is non-finite")
    expected_shape_prefix = [
        int(workload.get("sequence_length", -1)),
        int(workload.get("batch_size", -1)),
    ]
    output_shape = output.get("shape")
    if not isinstance(output_shape, list) or output_shape[:2] != expected_shape_prefix:
        raise ValueError(
            f"Output shape {output_shape!r} does not match logical workload "
            f"prefix {expected_shape_prefix!r}"
        )
    capabilities = record.get("telemetry_capabilities")
    memory = record.get("memory")
    if not isinstance(capabilities, Mapping) or not isinstance(memory, Mapping):
        raise ValueError("Record is missing telemetry capability or memory metadata")
    capability_to_field = {
        "process_rss": "process_rss_bytes",
        "backend_active_memory": "backend_active_bytes",
        "backend_peak_memory": "backend_peak_bytes",
    }
    for capability, field in capability_to_field.items():
        available = capabilities.get(capability)
        value = memory.get(field)
        if not isinstance(available, bool):
            raise ValueError(f"Telemetry capability '{capability}' must be boolean")
        if available and (not isinstance(value, int) or value < 0):
            raise ValueError(f"Capability '{capability}' requires memory field '{field}'")
        if not available and value is not None:
            raise ValueError(f"Unavailable capability '{capability}' must record '{field}' as null")
    _walk_finite(record)
    record_id = record.get("record_id")
    if record_id is not None:
        if not isinstance(record_id, str) or len(record_id) != 64:
            raise ValueError("record_id must be a SHA-256 hex digest")
        content = dict(record)
        del content["record_id"]
        expected = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if record_id != expected:
            raise ValueError("record_id does not match benchmark record content")


def write_benchmark_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    validate_benchmark_record(record)
    output = Path(path).expanduser().resolve()
    if output.exists() and not replace:
        raise FileExistsError(
            f"Authoritative benchmark record already exists: {output}. "
            "Pass replace=True or --replace to replace it explicitly."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Benchmarking requires pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return value


def _infer_model_name(path: Path, cfg: Mapping[str, Any]) -> str:
    if cfg.get("model_name"):
        model_name = str(cfg["model_name"]).strip().lower()
    elif "n_heads" in cfg and "ff_mult" in cfg:
        model_name = "transformer"
    elif "cell_type" in cfg:
        model_name = "neo"
    else:
        model_name = "lstm"
    if model_name not in {"neo", "lstm", "transformer"}:
        raise ValueError(f"Unsupported model_name '{model_name}' in {path}")
    return model_name


_CHECKPOINT_CONTRACT_KEYS = (
    "model_name",
    "vocab_size",
    "d_model",
    "d_embed",
    "n_layers",
    "n_heads",
    "ff_mult",
    "dropout",
    "tie_embeddings",
    "activation_id",
    "activation_sparsity_eps",
    "cell_type",
    "recurrent_norm",
    "recurrent_norm_place",
    "output_norm",
    "norm_place",
    "rmsnorm_eps",
    "lstm_bias_mode",
    "lstm_layer_dropout",
    "transformer_variant",
    "reference_backend",
    "use_checkpoint",
    "use_compile",
    "lr",
    "betas",
    "adam_eps",
    "grad_clip",
    "weight_decay",
    "weight_decay_policy",
    "embed_weight_decay",
    "recurrent_weight_decay",
    "proj_weight_decay",
    "transformer_weight_decay",
    "cosine",
    "epochs",
    "warmup_epochs",
    "min_lr",
    "train_regime",
    "stream_state",
    "tbptt_len",
    "block_size",
    "forget_bias_init",
    "recurrent_init",
)


def _validate_checkpoint_config(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    checkpoint_cfg = payload.get("cfg")
    if not isinstance(checkpoint_cfg, Mapping):
        raise ValueError("Checkpoint is missing config metadata required for benchmarking")
    conflicts = []
    for key in _CHECKPOINT_CONTRACT_KEYS:
        if key not in checkpoint_cfg or key not in cfg:
            continue
        left = checkpoint_cfg[key]
        right = cfg[key]
        if key == "rmsnorm_eps":
            matches = math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = left == right
        if not matches:
            conflicts.append(f"{key}: checkpoint={left!r}, config={right!r}")
    if conflicts:
        raise ValueError("Checkpoint metadata mismatch: " + "; ".join(conflicts))


def _checkpoint_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in ("format", "backend", "epoch", "global_step", "best_val")
        if key in payload
    } | {"config_snapshot": dict(payload.get("cfg", {}))}


def _logical_parameter_breakdown(
    entries: Iterable[tuple[str, int, bool]],
    model_name: str,
) -> Dict[str, int]:
    breakdown = {
        "embeddings": 0,
        "recurrent_or_transformer": 0,
        "projections_and_head": 0,
        "other": 0,
    }
    for name, size, trainable in entries:
        if not trainable:
            continue
        if name.startswith("emb.") or ".emb." in name or "pos_emb." in name:
            key = "embeddings"
        elif (
            (model_name == "neo" and "recurrent." in name)
            or (model_name == "lstm" and (name.startswith("lstm.") or name.startswith("lstm_layers.")))
            or (model_name == "transformer" and (name.startswith("blocks.") or name.startswith("encoder.")))
            or name.startswith("pre_norms.")
            or name.startswith("stack_norm.")
        ):
            key = "recurrent_or_transformer"
        elif name.startswith(("in_proj.", "out_proj.", "head.")) or name == "output_bias":
            key = "projections_and_head"
        else:
            key = "other"
        breakdown[key] += int(size)
    return breakdown


def _build_adapter(backend_name: str, model: Any, cfg: Mapping[str, Any], device: Any):
    if backend_name == "torch":
        from .backends.torch_efficiency import TorchEfficiencyAdapter

        return TorchEfficiencyAdapter(model, cfg, device)
    if backend_name == "mlx":
        from .backends.mlx_efficiency import MlxEfficiencyAdapter

        return MlxEfficiencyAdapter(model, cfg, device)
    raise ValueError(f"Unsupported benchmark backend '{backend_name}'")


def benchmark_from_paths(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    backend_name: str,
    device: str,
    workload: str,
    timing_scope: str,
    batch_size: int,
    sequence_length: int,
    warmup_iterations: int,
    measured_iterations: int,
    repetition_id: str,
    profile_label: str,
    weight_provenance: str,
    output_path: str | Path,
    seed: int,
    dry_run: bool = False,
    replace: bool = False,
) -> Dict[str, Any]:
    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if output in {config, checkpoint}:
        raise ValueError("Benchmark output path must differ from config and checkpoint paths")
    if output.exists() and not replace:
        raise FileExistsError(
            f"Authoritative benchmark record already exists: {output}. "
            "Pass replace=True or --replace to replace it explicitly."
        )
    backend_name = str(backend_name).strip().lower()
    if backend_name not in {"torch", "mlx"}:
        raise ValueError("backend must be 'torch' or 'mlx'")
    if not str(device).strip() or str(device).strip().lower() == "auto":
        raise ValueError("Benchmark device must be explicit; 'auto' is not accepted")
    if weight_provenance not in SUPPORTED_WEIGHT_PROVENANCE:
        raise ValueError(
            f"Unsupported weight_provenance '{weight_provenance}'. Expected one of: "
            f"{sorted(SUPPORTED_WEIGHT_PROVENANCE)}"
        )
    if not str(profile_label).strip():
        raise ValueError("profile_label must be non-empty")

    cfg = _load_yaml(config)
    if bool(cfg.get("use_checkpoint", False)):
        raise ValueError("Efficiency benchmark requires use_checkpoint=false")
    tbptt_len = int(cfg.get("tbptt_len", 0))
    if workload == "train_step" and 0 < tbptt_len < int(sequence_length):
        raise ValueError(
            "The isolated train_step benchmark requires full-sequence backpropagation; "
            f"config tbptt_len={tbptt_len} is shorter than sequence_length={sequence_length}"
        )
    model_name = _infer_model_name(config, cfg)
    payload = load_checkpoint_payload(checkpoint, map_location="cpu")
    _validate_checkpoint_config(payload, cfg)
    checkpoint_backend = infer_checkpoint_backend(payload)
    if weight_provenance == "backend_native_checkpoint" and checkpoint_backend != backend_name:
        raise ValueError(
            "backend_native_checkpoint provenance requires checkpoint backend to match "
            f"benchmark backend: checkpoint={checkpoint_backend}, benchmark={backend_name}"
        )

    backend = get_backend(backend_name)
    requested_device = str(device).strip().lower()
    if backend_name == "torch":
        import torch

        device_type = requested_device.split(":", 1)[0]
        if device_type not in {"cpu", "mps", "cuda"}:
            raise ValueError(
                f"Unsupported Torch benchmark device '{requested_device}'. Use cpu|mps|cuda."
            )
        if device_type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS benchmark requested but torch.backends.mps.is_available() is false")
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
    elif requested_device not in {"cpu", "gpu", "mps"}:
        raise ValueError(
            f"Unsupported MLX benchmark device '{requested_device}'. Use cpu|gpu|mps."
        )
    def execute_on_requested_device() -> Dict[str, Any]:
        runtime_device = backend.get_runtime_device(device)
        model = backend.build_model(cfg, model_name)
        if backend_name == "torch":
            model = model.to(runtime_device)
        backend.load_checkpoint_entry(
            checkpoint,
            model,
            device=runtime_device,
            cfg=cfg,
        )
        adapter = _build_adapter(backend_name, model, cfg, runtime_device)
        adapter.requested_device = str(device).strip().lower()
        parameter_entries = adapter.parameter_entries()
        parameter_breakdown = _logical_parameter_breakdown(parameter_entries, model_name)
        trainable_parameters = sum(parameter_breakdown.values())
        expected_parameters = int(backend.count_params(model))
        if trainable_parameters != expected_parameters:
            raise ValueError(
                "Parameter breakdown does not match backend trainable count: "
                f"breakdown={trainable_parameters}, backend={expected_parameters}"
            )

        root = Path(__file__).resolve().parents[2]
        metadata = {
            "identity": {
                "git_commit": _git_commit(root),
                "config_path": str(config),
                "config_sha256": _sha256_file(config),
                "config_snapshot": cfg,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "checkpoint_backend": checkpoint_backend,
                "checkpoint_metadata": _checkpoint_metadata(payload),
            },
            "model": {
                "name": model_name,
                "profile_label": str(profile_label),
                "weight_provenance": weight_provenance,
                "trainable_parameters": trainable_parameters,
                "parameter_breakdown": parameter_breakdown,
                "activation_id": cfg.get("activation_id"),
                "recurrent_norm": cfg.get("recurrent_norm", cfg.get("output_norm")),
                "recurrent_norm_place": cfg.get(
                    "recurrent_norm_place",
                    cfg.get("norm_place"),
                ),
                "use_checkpoint": bool(cfg.get("use_checkpoint", False)),
            },
        }
        spec = BenchmarkSpec(
            workload=workload,
            timing_scope=timing_scope,
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
            warmup_iterations=int(warmup_iterations),
            measured_iterations=int(measured_iterations),
            repetition_id=str(repetition_id),
            seed=int(seed),
            dry_run=bool(dry_run),
        )
        record = run_benchmark(adapter, spec, metadata)
        write_benchmark_record(output, record, replace=replace)
        return record

    if backend_name != "mlx":
        return execute_on_requested_device()

    import mlx.core as mx

    original_device = mx.default_device()
    try:
        return execute_on_requested_device()
    finally:
        mx.set_default_device(original_device)
