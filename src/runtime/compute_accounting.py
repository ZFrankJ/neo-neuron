"""Backend-neutral, shape-derived operation accounting for Neo and LSTM."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPUTE_SCHEMA_VERSION = "neo_manual_compute_v1"
DERIVED_REPORT_SCHEMA_VERSION = "neo_efficiency_derived_v1"
SUPPORTED_MODELS = {"neo", "lstm"}
SUPPORTED_WORKLOADS = {"train_step", "sequence_eval", "streaming_decode"}


@dataclass(frozen=True)
class ParameterEntry:
    name: str
    shape: tuple[int, ...]
    trainable: bool

    @property
    def size(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= int(dimension)
        return total


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> int:
    if key not in cfg:
        raise ValueError(f"Missing required compute-accounting config key: '{key}'")
    value = int(cfg[key])
    if value <= 0:
        raise ValueError(f"Compute-accounting config key '{key}' must be positive")
    return value


def parameter_entries_from_model(model: Any, backend: str) -> list[ParameterEntry]:
    """Expose parameter trees without putting compute formulas in backend adapters."""

    backend = str(backend).strip().lower()
    if backend == "torch":
        if not hasattr(model, "named_parameters"):
            raise TypeError("Torch compute accounting requires model.named_parameters()")
        return [
            ParameterEntry(
                str(name),
                tuple(int(value) for value in parameter.shape),
                bool(parameter.requires_grad),
            )
            for name, parameter in model.named_parameters()
        ]
    if backend == "mlx":
        try:
            from mlx.utils import tree_flatten
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("MLX compute accounting requested but MLX is unavailable") from exc
        return [
            ParameterEntry(
                str(name),
                tuple(int(value) for value in parameter.shape),
                True,
            )
            for name, parameter in tree_flatten(model.parameters())
        ]
    raise ValueError(f"Unsupported compute-accounting backend '{backend}'")


def _normalize_parameter_name(name: str, model_name: str) -> str | None:
    common = {
        "emb.weight": "embedding.weight",
        "in_proj.weight": "input_projection.weight",
        "in_proj.bias": "input_projection.bias",
        "out_proj.weight": "output_projection.weight",
        "out_proj.bias": "output_projection.bias",
        "head.weight": "output_head.weight",
        "head.bias": "output_head.bias",
        "output_bias": "output_head.bias",
    }
    if name in common:
        return common[name]

    if model_name == "neo":
        match = re.fullmatch(r"recurrent\.layers\.(\d+)\.fg_linear\.(weight|bias)", name)
        if match:
            return f"recurrent.layer.{match.group(1)}.affine.{match.group(2)}"
        match = re.fullmatch(r"recurrent\.pre_norms\.(\d+)\.(weight|bias)", name)
        if match:
            return f"recurrent.layer.{match.group(1)}.pre_norm.{match.group(2)}"
        match = re.fullmatch(r"recurrent\.stack_norm\.(weight|bias)", name)
        if match:
            return f"recurrent.stack_norm.{match.group(1)}"
        return None

    torch_lstm = (
        (r"lstm\.weight_ih_l(\d+)", "input.weight"),
        (r"lstm\.weight_hh_l(\d+)", "hidden.weight"),
        (r"lstm\.bias_ih_l(\d+)", "bias.input"),
        (r"lstm\.bias_hh_l(\d+)", "bias.hidden"),
    )
    for pattern, suffix in torch_lstm:
        match = re.fullmatch(pattern, name)
        if match:
            return f"recurrent.layer.{match.group(1)}.{suffix}"
    mlx_lstm = (
        (r"lstm_layers\.(\d+)\.Wx", "input.weight"),
        (r"lstm_layers\.(\d+)\.Wh", "hidden.weight"),
        (r"lstm_layers\.(\d+)\.bias", "bias.effective"),
    )
    for pattern, suffix in mlx_lstm:
        match = re.fullmatch(pattern, name)
        if match:
            return f"recurrent.layer.{match.group(1)}.{suffix}"
    match = re.fullmatch(r"(?:lstm\.)?pre_norms\.(\d+)\.(weight|bias)", name)
    if match:
        return f"recurrent.layer.{match.group(1)}.pre_norm.{match.group(2)}"
    match = re.fullmatch(r"(?:lstm\.)?stack_norm\.(weight|bias)", name)
    if match:
        return f"recurrent.stack_norm.{match.group(1)}"
    return None


def _norm_kind(cfg: Mapping[str, Any]) -> str:
    norm = str(cfg.get("recurrent_norm", cfg.get("output_norm", "layernorm"))).strip().lower()
    aliases = {
        "layer_norm": "layernorm",
        "ln": "layernorm",
        "rms_norm": "rmsnorm",
        "rms": "rmsnorm",
        "off": "none",
        "identity": "none",
    }
    norm = aliases.get(norm, norm)
    if norm not in {"layernorm", "rmsnorm", "none"}:
        raise ValueError(f"Unsupported normalization for compute accounting: '{norm}'")
    return norm


def _norm_place(cfg: Mapping[str, Any]) -> str:
    place = str(cfg.get("recurrent_norm_place", cfg.get("norm_place", "all"))).strip().lower()
    if place not in {"all", "pre", "stack"}:
        raise ValueError(f"Unsupported normalization placement for compute accounting: '{place}'")
    return place


def _expected_parameter_shapes(
    cfg: Mapping[str, Any],
    *,
    model_name: str,
    backend: str,
) -> dict[str, tuple[int, ...]]:
    vocab = _require_positive_int(cfg, "vocab_size")
    d_model = _require_positive_int(cfg, "d_model")
    d_embed = int(cfg.get("d_embed", d_model))
    n_layers = int(cfg.get("n_layers", 1))
    tie_embeddings = bool(cfg.get("tie_embeddings", True))
    norm = _norm_kind(cfg)
    place = _norm_place(cfg)

    expected: dict[str, tuple[int, ...]] = {"embedding.weight": (vocab, d_embed)}
    if d_embed != d_model:
        expected |= {
            "input_projection.weight": (d_model, d_embed),
            "input_projection.bias": (d_model,),
            "output_projection.weight": (d_embed, d_model),
            "output_projection.bias": (d_embed,),
        }
    norm_suffixes = (
        ("weight", "bias")
        if norm == "layernorm"
        else (("weight",) if norm == "rmsnorm" else ())
    )
    for layer in range(n_layers):
        if model_name == "neo":
            expected[f"recurrent.layer.{layer}.affine.weight"] = (2 * d_model, d_model)
            expected[f"recurrent.layer.{layer}.affine.bias"] = (2 * d_model,)
        else:
            expected[f"recurrent.layer.{layer}.input.weight"] = (4 * d_model, d_model)
            expected[f"recurrent.layer.{layer}.hidden.weight"] = (4 * d_model, d_model)
            if backend == "torch":
                expected[f"recurrent.layer.{layer}.bias.input"] = (4 * d_model,)
                expected[f"recurrent.layer.{layer}.bias.hidden"] = (4 * d_model,)
            else:
                expected[f"recurrent.layer.{layer}.bias.effective"] = (4 * d_model,)
        if norm != "none" and place in {"all", "pre"}:
            for suffix in norm_suffixes:
                expected[f"recurrent.layer.{layer}.pre_norm.{suffix}"] = (d_model,)
    if norm != "none" and place in {"all", "stack"}:
        for suffix in norm_suffixes:
            expected[f"recurrent.stack_norm.{suffix}"] = (d_model,)
    if tie_embeddings:
        expected["output_head.bias"] = (vocab,)
    else:
        expected["output_head.weight"] = (vocab, d_model)
        expected["output_head.bias"] = (vocab,)
    return expected


def _component_for_parameter(logical_name: str) -> str:
    if logical_name.startswith("embedding."):
        return "embedding_lookup"
    if logical_name.startswith("input_projection."):
        return "input_projection"
    if ".pre_norm." in logical_name:
        return "recurrent_pre_norm"
    if logical_name.startswith("recurrent.stack_norm."):
        return "recurrent_stack_norm"
    if logical_name.startswith("recurrent.layer."):
        return "recurrent_core"
    if logical_name.startswith("output_projection."):
        return "output_projection"
    if logical_name.startswith("output_head."):
        return "output_head"
    raise ValueError(f"Unknown logical parameter '{logical_name}'")


def _validate_parameter_coverage(
    entries: Sequence[ParameterEntry],
    cfg: Mapping[str, Any],
    *,
    model_name: str,
    backend: str,
    output_projection_executed: bool,
) -> tuple[dict[str, Any], int]:
    expected = _expected_parameter_shapes(cfg, model_name=model_name, backend=backend)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    unknown_nontrainable: list[str] = []
    active_trainable = 0
    covered_components: set[str] = set()

    for entry in entries:
        logical_name = _normalize_parameter_name(entry.name, model_name)
        if logical_name is None:
            if entry.trainable:
                raise ValueError(f"Unknown trainable parameter for compute coverage: {entry.name}")
            unknown_nontrainable.append(entry.name)
            continue
        if logical_name not in expected:
            raise ValueError(
                "Parameter is incompatible with the declared compute contract: "
                f"{entry.name} -> {logical_name}"
            )
        if tuple(entry.shape) != expected[logical_name]:
            raise ValueError(
                f"Parameter shape mismatch for {entry.name}: expected "
                f"{expected[logical_name]}, got {tuple(entry.shape)}"
            )
        if logical_name in seen:
            raise ValueError(f"Duplicate logical parameter coverage for {logical_name}")
        seen.add(logical_name)
        component = _component_for_parameter(logical_name)
        covered_components.add(component)
        executed = component != "output_projection" or output_projection_executed
        if entry.trainable and executed:
            active_trainable += entry.size
        normalized.append(
            {
                "backend_name": entry.name,
                "logical_name": logical_name,
                "shape": list(entry.shape),
                "trainable": bool(entry.trainable),
                "executed": executed,
            }
        )

    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"Missing parameter coverage for declared model components: {missing}")
    coverage = {
        "backend": backend,
        "covered_logical_components": sorted(covered_components),
        "normalized_parameters": sorted(normalized, key=lambda value: value["logical_name"]),
        "uncovered_trainable_parameters": [],
        "uncovered_nontrainable_parameters": sorted(unknown_nontrainable),
    }
    return coverage, active_trainable


def _manifest_entry(
    component: str,
    formula: str,
    macs: int,
    *,
    executed: bool = True,
    parameterized: bool = False,
) -> dict[str, Any]:
    return {
        "component": component,
        "executed": bool(executed),
        "parameterized": bool(parameterized),
        "formula": formula,
        "macs": int(macs),
    }


def _activation_category(activation_id: Any) -> str:
    text = str(activation_id if activation_id is not None else "id3").strip().lower()
    if text.startswith("id"):
        text = text[2:]
    if text in {"100", "tanh"}:
        return "tanh"
    if text in {"102", "none", "identity"}:
        return "none"
    if text in {"103", "gain_tanh", "one_plus_tanh", "1+tanh"}:
        return "gain_tanh"
    if text in {"3", "4", "5", "101", "gelu"}:
        return "activation"
    raise ValueError(f"Unsupported Neo activation for compute accounting: '{activation_id}'")


def account_model_operations(
    model: Any,
    cfg: Mapping[str, Any],
    *,
    backend: str,
    workload: str,
    batch_size: int,
    sequence_length: int,
    parameter_entries: Sequence[ParameterEntry] | None = None,
) -> dict[str, Any]:
    """Audit one logical recurrent workload and fail closed on parameter gaps."""

    model_name = str(cfg.get("model_name", "")).strip().lower()
    backend = str(backend).strip().lower()
    workload = str(workload).strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Manual compute accounting supports only {sorted(SUPPORTED_MODELS)}")
    if workload not in SUPPORTED_WORKLOADS:
        raise ValueError(f"Unsupported compute-accounting workload '{workload}'")
    if backend not in {"torch", "mlx"}:
        raise ValueError(f"Unsupported compute-accounting backend '{backend}'")
    batch_size = int(batch_size)
    sequence_length = int(sequence_length)
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")

    vocab = _require_positive_int(cfg, "vocab_size")
    d_model = _require_positive_int(cfg, "d_model")
    d_embed = int(cfg.get("d_embed", d_model))
    n_layers = int(cfg.get("n_layers", 1))
    if d_embed <= 0 or n_layers <= 0:
        raise ValueError("d_embed and n_layers must be positive")
    tie_embeddings = bool(cfg.get("tie_embeddings", True))
    if model_name == "neo" and str(cfg.get("cell_type", "cortical")).strip().lower() != "cortical":
        raise ValueError("Manual compute accounting supports only the cortical Neo cell")
    if model_name == "lstm":
        bias_mode = str(cfg.get("lstm_bias_mode", "split")).strip().lower()
        if bias_mode not in {"single", "split"}:
            raise ValueError(f"Unsupported LSTM bias mode for compute accounting: '{bias_mode}'")
    norm = _norm_kind(cfg)
    place = _norm_place(cfg)
    tokens = batch_size * sequence_length
    training = workload == "train_step"
    timing_scope = "end_to_end_loop" if training else "model_only"
    output_projection_executed = tie_embeddings and d_embed != d_model

    non_matmul = {
        "activation_elements": 0,
        "bias_additions": 0,
        "dropout_elements": 0,
        "elementwise_operations": 0,
        "loss_elements": 0,
        "normalization_elements": 0,
        "sigmoid_elements": 0,
        "softmax_elements": 0,
        "tanh_elements": 0,
    }
    manifest: list[dict[str, Any]] = [
        _manifest_entry(
            "embedding_lookup",
            "tokens * d_embed lookup elements; embedding lookup is data movement",
            0,
            parameterized=True,
        )
    ]
    macs = 0

    if d_embed != d_model:
        projection_macs = tokens * d_embed * d_model
        macs += projection_macs
        non_matmul["bias_additions"] += tokens * d_model
        manifest.append(
            _manifest_entry(
                "input_projection",
                "tokens * d_embed * d_model",
                projection_macs,
                parameterized=True,
            )
        )
    else:
        manifest.append(_manifest_entry("input_projection", "identity", 0))

    pre_norms = n_layers if norm != "none" and place in {"all", "pre"} else 0
    stack_norms = 1 if norm != "none" and place in {"all", "stack"} else 0
    non_matmul["normalization_elements"] += tokens * d_model * (pre_norms + stack_norms)
    manifest.append(
        _manifest_entry(
            "recurrent_pre_norm",
            "tokens * d_model * normalized_layers",
            0,
            executed=pre_norms > 0,
            parameterized=pre_norms > 0,
        )
    )

    if model_name == "neo":
        recurrent_macs = tokens * n_layers * d_model * (2 * d_model)
        macs += recurrent_macs
        non_matmul["bias_additions"] += tokens * n_layers * 2 * d_model
        non_matmul["elementwise_operations"] += tokens * n_layers * 2 * d_model
        activation = _activation_category(cfg.get("activation_id", "id3"))
        if activation == "tanh":
            non_matmul["tanh_elements"] += tokens * n_layers * d_model
        elif activation == "gain_tanh":
            non_matmul["tanh_elements"] += tokens * n_layers * d_model
            non_matmul["elementwise_operations"] += tokens * n_layers * d_model
        elif activation == "activation":
            non_matmul["activation_elements"] += tokens * n_layers * d_model
        manifest.append(
            _manifest_entry(
                "recurrent_core",
                "tokens * n_layers * d_model * (2 * d_model)",
                recurrent_macs,
                parameterized=True,
            )
        )
        manifest.append(
            _manifest_entry(
                "recurrent_state_update",
                "one declared activation and state/output elementwise formula per recurrent element",
                0,
            )
        )
    else:
        recurrent_macs = tokens * n_layers * 8 * d_model * d_model
        macs += recurrent_macs
        non_matmul["bias_additions"] += tokens * n_layers * 4 * d_model
        non_matmul["elementwise_operations"] += tokens * n_layers * 8 * d_model
        non_matmul["sigmoid_elements"] += tokens * n_layers * 3 * d_model
        non_matmul["tanh_elements"] += tokens * n_layers * 2 * d_model
        manifest.append(
            _manifest_entry(
                "recurrent_core",
                "tokens * n_layers * (4*d_model*d_model input + 4*d_model*d_model hidden)",
                recurrent_macs,
                parameterized=True,
            )
        )
        manifest.append(
            _manifest_entry(
                "recurrent_gate_update",
                "three sigmoid, two tanh, four gate/cell elementwise operations per hidden element",
                0,
            )
        )

    manifest.append(
        _manifest_entry(
            "recurrent_stack_norm",
            "tokens * d_model normalized elements",
            0,
            executed=stack_norms > 0,
            parameterized=stack_norms > 0,
        )
    )

    dropout = float(cfg.get("dropout", 0.0))
    if training and dropout > 0.0:
        non_matmul["dropout_elements"] += tokens * d_model
    lstm_inter_layer_dropout_executed = False
    if model_name == "lstm":
        layer_dropout = float(cfg.get("lstm_layer_dropout", dropout))
        lstm_inter_layer_dropout_executed = (
            training and layer_dropout > 0.0 and n_layers > 1
        )
        if lstm_inter_layer_dropout_executed:
            non_matmul["dropout_elements"] += tokens * d_model * (n_layers - 1)
    manifest.append(
        _manifest_entry(
            "output_dropout",
            "tokens * d_model when train_step dropout is nonzero",
            0,
            executed=training and dropout > 0.0,
        )
    )
    if model_name == "lstm":
        manifest.append(
            _manifest_entry(
                "lstm_inter_layer_dropout",
                (
                    "tokens * d_model * (n_layers - 1) when train_step "
                    "lstm_layer_dropout is nonzero"
                ),
                0,
                executed=lstm_inter_layer_dropout_executed,
            )
        )

    if d_embed != d_model:
        projection_macs = tokens * d_model * d_embed if output_projection_executed else 0
        macs += projection_macs
        if output_projection_executed:
            non_matmul["bias_additions"] += tokens * d_embed
        manifest.append(
            _manifest_entry(
                "output_projection",
                "tokens * d_model * d_embed" if output_projection_executed else "unused_with_untied_head",
                projection_macs,
                executed=output_projection_executed,
                parameterized=True,
            )
        )
    else:
        manifest.append(_manifest_entry("output_projection", "identity", 0))

    head_input = d_embed if tie_embeddings else d_model
    head_macs = tokens * head_input * vocab
    macs += head_macs
    non_matmul["bias_additions"] += tokens * vocab
    manifest.append(
        _manifest_entry(
            "output_head",
            (
                "tokens * d_embed * vocab_size using tied embedding weight"
                if tie_embeddings
                else "tokens * d_model * vocab_size"
            ),
            head_macs,
            parameterized=True,
        )
    )

    if training:
        non_matmul["softmax_elements"] = tokens * vocab
        non_matmul["loss_elements"] = tokens
    manifest.append(
        _manifest_entry(
            "cross_entropy",
            "tokens * vocab_size softmax elements plus tokens loss elements",
            0,
            executed=training,
        )
    )

    entries = list(parameter_entries or parameter_entries_from_model(model, backend))
    parameter_coverage, active_trainable = _validate_parameter_coverage(
        entries,
        cfg,
        model_name=model_name,
        backend=backend,
        output_projection_executed=output_projection_executed,
    )
    backward = (
        {
            "status": "estimated",
            "basis": "2x_forward_parameterized_dense_macs",
            "macs": 2 * macs,
            "flops": 4 * macs,
            "non_matmul_operations": None,
        }
        if training
        else {
            "status": "not_executed",
            "basis": None,
            "macs": 0,
            "flops": 0,
            "non_matmul_operations": None,
        }
    )
    optimizer = (
        {
            "status": "estimated",
            "algorithm": "adamw",
            "updated_parameter_elements": active_trainable,
            "gradient_clipping_parameter_elements": (
                active_trainable if float(cfg.get("grad_clip", 1.0)) > 0.0 else 0
            ),
            "primitive_scalar_operations": None,
        }
        if training
        else {
            "status": "not_executed",
            "algorithm": None,
            "updated_parameter_elements": 0,
            "gradient_clipping_parameter_elements": 0,
            "primitive_scalar_operations": None,
        }
    )
    logical_operations = {
        "model": {
            "name": model_name,
            "vocab_size": vocab,
            "d_model": d_model,
            "d_embed": d_embed,
            "n_layers": n_layers,
            "tie_embeddings": tie_embeddings,
            "recurrent_norm": norm,
            "recurrent_norm_place": place,
            "activation_id": cfg.get("activation_id") if model_name == "neo" else None,
        },
        "workload": {
            "name": workload,
            "timing_scope": timing_scope,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "tokens": tokens,
            "tokens_per_iteration": tokens,
        },
        "conventions": {
            "mac_to_flop": "1 MAC = 2 FLOPs",
            "non_matmul_count_unit": (
                "tensor elements evaluated by each declared category, not "
                "matmul-equivalent primitive costs"
            ),
            "forward_status": "exact_under_coverage_manifest",
            "backward_status": "estimated_unless_derivatives_are_enumerated",
            "optimizer_status": "estimated_parameter_elements_not_primitive_ops",
            "kernel_fusion_or_hardware_utilization_included": False,
        },
        "coverage_manifest": manifest,
        "data_movement": {
            "embedding_lookup_elements": tokens * d_embed,
            "embedding_lookup_macs": 0,
        },
        "forward": {
            "status": "exact",
            "macs": macs,
            "flops": 2 * macs,
            "non_matmul_operations": non_matmul,
        },
        "backward": backward,
        "optimizer": optimizer,
    }
    record: dict[str, Any] = {
        "schema_version": COMPUTE_SCHEMA_VERSION,
        "logical_operations": logical_operations,
        "parameter_coverage": parameter_coverage,
    }
    record["operation_record_id"] = _canonical_hash(record)
    return record


def _workload_identity(workload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    fields = {
        key: workload.get(key)
        for key in (
            "name",
            "timing_scope",
            "batch_size",
            "sequence_length",
            "tokens_per_iteration",
        )
    }
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        raise ValueError(f"Benchmark workload is missing immutable identity fields: {missing}")
    name = str(fields["name"])
    if name not in SUPPORTED_WORKLOADS:
        raise ValueError(f"Unsupported workload in immutable identity: {name!r}")
    expected_scope = "end_to_end_loop" if name == "train_step" else "model_only"
    if fields["timing_scope"] != expected_scope:
        raise ValueError(
            f"Workload {name!r} requires timing_scope {expected_scope!r}, "
            f"got {fields['timing_scope']!r}"
        )
    batch_size = int(fields["batch_size"])
    sequence_length = int(fields["sequence_length"])
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Workload batch_size and sequence_length must be positive")
    expected_tokens = batch_size * sequence_length
    if int(fields["tokens_per_iteration"]) != expected_tokens:
        raise ValueError("Workload tokens_per_iteration does not match its logical shape")
    if "tokens" in workload and int(workload["tokens"]) != expected_tokens:
        raise ValueError("Logical operation workload tokens do not match its logical shape")
    fields = {
        "name": name,
        "timing_scope": expected_scope,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "tokens_per_iteration": expected_tokens,
    }
    return _canonical_hash(fields), fields


def operation_record_from_benchmark(benchmark_record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive an audited operation record without modifying benchmark evidence."""

    from .efficiency import validate_benchmark_record

    validate_benchmark_record(benchmark_record)

    identity = benchmark_record.get("identity")
    runtime = benchmark_record.get("runtime")
    model_metadata = benchmark_record.get("model")
    workload = benchmark_record.get("workload")
    if not all(isinstance(value, Mapping) for value in (identity, runtime, model_metadata, workload)):
        raise ValueError("Benchmark record is missing identity, runtime, model, or workload metadata")
    assert isinstance(identity, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(model_metadata, Mapping)
    assert isinstance(workload, Mapping)
    benchmark_id = str(benchmark_record.get("record_id", "")).strip()
    config_sha = str(identity.get("config_sha256", "")).strip()
    checkpoint_sha = str(identity.get("checkpoint_sha256", "")).strip()
    cfg = identity.get("effective_config_snapshot")
    backend = str(runtime.get("backend", "")).strip().lower()
    model_name = str(model_metadata.get("name", "")).strip().lower()
    if not benchmark_id or not config_sha or not checkpoint_sha or not isinstance(cfg, Mapping):
        raise ValueError("Benchmark record is missing immutable identity or effective config")
    if str(cfg.get("model_name", "")).strip().lower() != model_name:
        raise ValueError("Benchmark model name does not match effective config")
    workload_id, _ = _workload_identity(workload)

    from .backend import get_backend

    backend_module = get_backend(backend)

    def build_and_account() -> dict[str, Any]:
        model = backend_module.build_model(dict(cfg), model_name)
        return account_model_operations(
            model,
            cfg,
            backend=backend,
            workload=str(workload["name"]),
            batch_size=int(workload["batch_size"]),
            sequence_length=int(workload["sequence_length"]),
        )

    if backend != "mlx":
        record = build_and_account()
    else:
        import mlx.core as mx

        original_device = mx.default_device()
        try:
            mx.set_default_device(mx.cpu)
            record = build_and_account()
        finally:
            mx.set_default_device(original_device)
    logical_workload_id, _ = _workload_identity(
        record["logical_operations"]["workload"]
    )
    if logical_workload_id != workload_id:
        raise ValueError(
            "Generated logical operation workload does not match benchmark workload_id"
        )
    record["identity"] = {
        "benchmark_record_id": benchmark_id,
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "workload_id": workload_id,
    }
    record.pop("operation_record_id", None)
    record["operation_record_id"] = _canonical_hash(record)
    return record


def derive_efficiency_report(
    benchmark_record: Mapping[str, Any],
    operation_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Join timing and arithmetic only after all immutable identifiers match."""

    from .efficiency import validate_benchmark_record

    validate_benchmark_record(benchmark_record)

    benchmark_identity = benchmark_record.get("identity")
    operation_identity = operation_record.get("identity")
    workload = benchmark_record.get("workload")
    if not isinstance(benchmark_identity, Mapping) or not isinstance(operation_identity, Mapping):
        raise ValueError("Both records require immutable identity metadata")
    if not isinstance(workload, Mapping):
        raise ValueError("Benchmark record is missing workload metadata")
    workload_id, _ = _workload_identity(workload)
    expected = {
        "benchmark_record_id": str(benchmark_record.get("record_id", "")),
        "config_sha256": str(benchmark_identity.get("config_sha256", "")),
        "checkpoint_sha256": str(benchmark_identity.get("checkpoint_sha256", "")),
        "workload_id": workload_id,
    }
    for key, value in expected.items():
        if operation_identity.get(key) != value:
            raise ValueError(
                f"Cannot derive efficiency report: {key} mismatch "
                f"({operation_identity.get(key)!r} != {value!r})"
            )
    _validate_compute_record(operation_record)
    operation_id = str(operation_record.get("operation_record_id", "")).strip()
    if not operation_id:
        raise ValueError("Operation record is missing operation_record_id")
    summary = benchmark_record.get("summary")
    benchmark_evidence = benchmark_record.get("evidence")
    logical = operation_record.get("logical_operations")
    if (
        not isinstance(summary, Mapping)
        or not isinstance(benchmark_evidence, Mapping)
        or not isinstance(logical, Mapping)
    ):
        raise ValueError("Timing summary, evidence, or logical operation summary is missing")
    report: dict[str, Any] = {
        "schema_version": DERIVED_REPORT_SCHEMA_VERSION,
        "identity": expected | {"operation_record_id": operation_id},
        "timing_summary": dict(summary),
        "evidence": {
            "status": benchmark_evidence["status"],
            "provisional_reasons": list(benchmark_evidence["provisional_reasons"]),
            "source_dry_run": bool(workload["dry_run"]),
        },
        "operation_summary": {
            key: logical[key]
            for key in ("forward", "backward", "optimizer")
            if key in logical
        },
        "interpretation": {
            "hardware_utilization_claim": False,
            "statement": (
                "Arithmetic counts describe the audited mathematical workload; "
                "timing describes the selected backend and device."
            ),
        },
    }
    report["derived_report_id"] = _canonical_hash(report)
    return report


def _validate_compute_record(record: Mapping[str, Any]) -> None:
    schema = record.get("schema_version")
    if schema == COMPUTE_SCHEMA_VERSION:
        id_key = "operation_record_id"
    elif schema == DERIVED_REPORT_SCHEMA_VERSION:
        id_key = "derived_report_id"
    else:
        raise ValueError(f"Unsupported compute record schema_version: {schema!r}")
    record_id = record.get(id_key)
    if not isinstance(record_id, str) or len(record_id) != 64:
        raise ValueError(f"{id_key} must be a SHA-256 hex digest")
    content = dict(record)
    del content[id_key]
    if record_id != _canonical_hash(content):
        raise ValueError(f"{id_key} does not match compute record content")
    if schema == COMPUTE_SCHEMA_VERSION:
        identity = record.get("identity")
        logical = record.get("logical_operations")
        if identity is None:
            return
        if not isinstance(identity, Mapping) or not isinstance(logical, Mapping):
            raise ValueError("Linked compute records require identity and logical_operations")
        workload = logical.get("workload")
        if not isinstance(workload, Mapping):
            raise ValueError("Linked compute record is missing its logical workload")
        logical_workload_id, _ = _workload_identity(workload)
        if identity.get("workload_id") != logical_workload_id:
            raise ValueError(
                "Compute record identity workload_id does not match logical_operations.workload"
            )
        return

    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "status",
        "provisional_reasons",
        "source_dry_run",
    }:
        raise ValueError("Derived reports require complete benchmark evidence metadata")
    reasons = evidence.get("provisional_reasons")
    if not isinstance(reasons, list) or any(not isinstance(value, str) for value in reasons):
        raise ValueError("Derived report provisional_reasons must be a list of strings")
    if evidence.get("source_dry_run") is True:
        if evidence.get("status") != "provisional" or "dry_run" not in reasons:
            raise ValueError("Dry-run derived reports must remain explicitly provisional")
    elif evidence.get("source_dry_run") is False:
        if evidence.get("status") != "authoritative" or reasons:
            raise ValueError("Formal derived reports must remain authoritative")
    else:
        raise ValueError("Derived report source_dry_run must be boolean")


def write_compute_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    _validate_compute_record(record)
    output = Path(path).expanduser().resolve()
    if output.exists() and not replace:
        raise FileExistsError(
            f"Authoritative compute record already exists: {output}. Pass replace=True to replace it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
