from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.runtime.backends import torch_backend
from src.runtime.efficiency import benchmark_from_paths
from src.runtime.compute_accounting import (
    COMPUTE_SCHEMA_VERSION,
    ParameterEntry,
    account_model_operations,
    derive_efficiency_report,
    operation_record_from_benchmark,
    parameter_entries_from_model,
    write_compute_record,
)


def _tiny_cfg(
    model_name: str,
    *,
    tie_embeddings: bool = True,
    d_embed: int = 2,
) -> dict:
    cfg = {
        "model_name": model_name,
        "vocab_size": 5,
        "d_model": 2,
        "d_embed": d_embed,
        "n_layers": 1,
        "dropout": 0.0,
        "tie_embeddings": tie_embeddings,
        "recurrent_norm": "rmsnorm",
        "recurrent_norm_place": "all",
        "rmsnorm_eps": 1e-5,
        "reference_backend": "mlx",
        "use_checkpoint": False,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
    }
    if model_name == "neo":
        cfg.update(
            {
                "cell_type": "cortical",
                "activation_id": "tanh",
                "weight_decay_policy": "table",
            }
        )
    else:
        cfg.update(
            {
                "lstm_bias_mode": "single",
                "lstm_layer_dropout": 0.0,
            }
        )
    return cfg


def _account_torch(cfg: dict, *, workload: str = "sequence_eval") -> dict:
    model = torch_backend.build_model(cfg, cfg["model_name"])
    return account_model_operations(
        model,
        cfg,
        backend="torch",
        workload=workload,
        batch_size=1,
        sequence_length=2,
    )


def test_tiny_neo_forward_counts_are_hand_computable_and_embedding_is_not_a_mac():
    record = _account_torch(_tiny_cfg("neo"))
    logical = record["logical_operations"]

    assert record["schema_version"] == COMPUTE_SCHEMA_VERSION
    assert logical["forward"] == {
        "status": "exact",
        "macs": 36,
        "flops": 72,
        "non_matmul_operations": {
            "activation_elements": 0,
            "bias_additions": 18,
            "dropout_elements": 0,
            "elementwise_operations": 8,
            "loss_elements": 0,
            "normalization_elements": 8,
            "sigmoid_elements": 0,
            "softmax_elements": 0,
            "tanh_elements": 4,
        },
    }
    assert logical["data_movement"] == {
        "embedding_lookup_elements": 4,
        "embedding_lookup_macs": 0,
    }
    assert logical["conventions"]["mac_to_flop"] == "1 MAC = 2 FLOPs"
    assert logical["conventions"]["forward_status"] == "exact_under_coverage_manifest"


def test_tiny_lstm_train_step_separates_exact_forward_from_estimates():
    record = _account_torch(_tiny_cfg("lstm"), workload="train_step")
    logical = record["logical_operations"]

    assert logical["forward"]["macs"] == 84
    assert logical["forward"]["flops"] == 168
    assert logical["forward"]["non_matmul_operations"] == {
        "activation_elements": 0,
        "bias_additions": 26,
        "dropout_elements": 0,
        "elementwise_operations": 32,
        "loss_elements": 2,
        "normalization_elements": 8,
        "sigmoid_elements": 12,
        "softmax_elements": 10,
        "tanh_elements": 8,
    }
    assert logical["backward"] == {
        "status": "estimated",
        "basis": "2x_forward_parameterized_dense_macs",
        "macs": 168,
        "flops": 336,
        "non_matmul_operations": None,
    }
    assert logical["optimizer"] == {
        "status": "estimated",
        "algorithm": "adamw",
        "updated_parameter_elements": 59,
        "gradient_clipping_parameter_elements": 59,
        "primitive_scalar_operations": None,
    }


@pytest.mark.parametrize(
    ("activation_id", "activation_elements", "tanh_elements", "elementwise_operations"),
    [
        ("id3", 4, 0, 8),
        ("gelu", 4, 0, 8),
        ("none", 0, 0, 8),
        ("gain_tanh", 0, 4, 12),
    ],
)
def test_supported_neo_activations_keep_separate_non_matmul_categories(
    activation_id: str,
    activation_elements: int,
    tanh_elements: int,
    elementwise_operations: int,
):
    cfg = _tiny_cfg("neo") | {"activation_id": activation_id}
    operations = _account_torch(cfg)["logical_operations"]["forward"][
        "non_matmul_operations"
    ]

    assert operations["activation_elements"] == activation_elements
    assert operations["tanh_elements"] == tanh_elements
    assert operations["elementwise_operations"] == elementwise_operations


@pytest.mark.parametrize(
    ("norm", "place", "expected_elements", "expected_parameterized_components"),
    [
        ("none", "all", 0, []),
        ("layernorm", "pre", 4, ["recurrent_pre_norm"]),
        ("rmsnorm", "stack", 4, ["recurrent_stack_norm"]),
    ],
)
def test_supported_normalization_settings_are_manifested(
    norm: str,
    place: str,
    expected_elements: int,
    expected_parameterized_components: list[str],
):
    cfg = _tiny_cfg("neo") | {
        "recurrent_norm": norm,
        "recurrent_norm_place": place,
    }
    record = _account_torch(cfg)
    logical = record["logical_operations"]
    parameterized = [
        entry["component"]
        for entry in logical["coverage_manifest"]
        if entry["parameterized"]
        and entry["component"] in {"recurrent_pre_norm", "recurrent_stack_norm"}
    ]

    assert logical["forward"]["non_matmul_operations"]["normalization_elements"] == (
        expected_elements
    )
    assert parameterized == expected_parameterized_components


def test_tied_and_untied_heads_cover_executed_and_unexecuted_projection_parameters():
    tied = _account_torch(_tiny_cfg("neo", tie_embeddings=True, d_embed=3))
    untied = _account_torch(_tiny_cfg("neo", tie_embeddings=False, d_embed=3))

    assert tied["logical_operations"]["forward"]["macs"] == 70
    assert untied["logical_operations"]["forward"]["macs"] == 48
    manifest = {
        entry["component"]: entry
        for entry in untied["logical_operations"]["coverage_manifest"]
    }
    assert manifest["output_projection"]["executed"] is False
    assert manifest["output_projection"]["formula"] == "unused_with_untied_head"
    assert manifest["output_head"]["formula"] == "tokens * d_model * vocab_size"
    assert untied["parameter_coverage"]["uncovered_trainable_parameters"] == []


@pytest.mark.parametrize("model_name", ["neo", "lstm"])
def test_unknown_trainable_parameter_fails_closed(model_name: str):
    cfg = _tiny_cfg(model_name)
    model = torch_backend.build_model(cfg, model_name)
    entries = parameter_entries_from_model(model, "torch")
    entries.append(ParameterEntry("mystery.weight", (2, 2), True))

    with pytest.raises(ValueError, match="Unknown trainable parameter.*mystery.weight"):
        account_model_operations(
            model,
            cfg,
            backend="torch",
            workload="sequence_eval",
            batch_size=1,
            sequence_length=2,
            parameter_entries=entries,
        )


@pytest.mark.parametrize("model_name", ["neo", "lstm"])
def test_torch_and_mlx_models_have_identical_logical_operation_records(model_name: str):
    mlx = pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    cfg = _tiny_cfg(model_name)
    torch_record = _account_torch(cfg)
    original_device = mlx.default_device()
    try:
        mlx.set_default_device(mlx.cpu)
        mlx_model = mlx_backend.build_model(cfg, model_name)
        mlx_record = account_model_operations(
            mlx_model,
            cfg,
            backend="mlx",
            workload="sequence_eval",
            batch_size=1,
            sequence_length=2,
        )
    finally:
        mlx.set_default_device(original_device)

    assert torch_record["logical_operations"] == mlx_record["logical_operations"]
    assert torch_record["parameter_coverage"]["covered_logical_components"] == (
        mlx_record["parameter_coverage"]["covered_logical_components"]
    )
    assert torch_record["parameter_coverage"]["uncovered_trainable_parameters"] == []
    assert mlx_record["parameter_coverage"]["uncovered_trainable_parameters"] == []


def _benchmark_record(tmp_path: Path) -> dict:
    import yaml

    cfg = _tiny_cfg("neo") | {"run_tag": "tiny-compute"}
    config_path = tmp_path / "tiny.yaml"
    checkpoint_path = tmp_path / "tiny.pt"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = torch_backend.build_model(cfg, "neo")
    torch_backend.save_checkpoint_entry(
        checkpoint_path,
        model,
        None,
        None,
        epoch=1,
        global_step=2,
        cfg=cfg,
    )
    return benchmark_from_paths(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        backend_name="torch",
        device="cpu",
        workload="sequence_eval",
        timing_scope="model_only",
        batch_size=1,
        sequence_length=2,
        warmup_iterations=0,
        measured_iterations=1,
        repetition_id="compute-link",
        profile_label=cfg["run_tag"],
        output_path=tmp_path / "benchmark.json",
        seed=17,
        dry_run=True,
    )


def _rehash_operation_record(record: dict) -> None:
    record.pop("operation_record_id", None)
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["operation_record_id"] = hashlib.sha256(canonical).hexdigest()


def test_compute_and_timing_join_only_through_immutable_derived_report_keys(tmp_path: Path):
    benchmark = _benchmark_record(tmp_path)
    operations = operation_record_from_benchmark(benchmark)
    report = derive_efficiency_report(benchmark, operations)

    assert operations["identity"] == {
        "benchmark_record_id": benchmark["record_id"],
        "config_sha256": benchmark["identity"]["config_sha256"],
        "checkpoint_sha256": benchmark["identity"]["checkpoint_sha256"],
        "workload_id": operations["identity"]["workload_id"],
    }
    assert report["identity"] == {
        "benchmark_record_id": benchmark["record_id"],
        "operation_record_id": operations["operation_record_id"],
        "config_sha256": benchmark["identity"]["config_sha256"],
        "checkpoint_sha256": benchmark["identity"]["checkpoint_sha256"],
        "workload_id": operations["identity"]["workload_id"],
    }
    assert report["interpretation"]["hardware_utilization_claim"] is False
    assert report["evidence"] == {
        "status": "provisional",
        "provisional_reasons": ["dry_run"],
        "source_dry_run": True,
    }

    mismatched = copy.deepcopy(operations)
    mismatched["identity"]["checkpoint_sha256"] = "other-checkpoint"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        derive_efficiency_report(benchmark, mismatched)

    tampered = copy.deepcopy(operations)
    tampered["logical_operations"]["forward"]["macs"] += 1
    with pytest.raises(ValueError, match="operation_record_id"):
        derive_efficiency_report(benchmark, tampered)

    rebound = copy.deepcopy(operations)
    rebound["logical_operations"]["workload"].update(
        {
            "sequence_length": 999,
            "tokens": 999,
            "tokens_per_iteration": 999,
        }
    )
    _rehash_operation_record(rebound)
    with pytest.raises(ValueError, match="workload_id"):
        derive_efficiency_report(benchmark, rebound)


def test_compute_record_writer_is_immutable_without_replace(tmp_path: Path):
    record = _account_torch(_tiny_cfg("neo"))
    output = tmp_path / "compute.json"

    write_compute_record(output, record)
    with pytest.raises(FileExistsError, match="replace"):
        write_compute_record(output, record)
    write_compute_record(output, record, replace=True)

    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_macos_efficiency_gate_includes_compute_accounting():
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/tests.yml").read_text(encoding="utf-8")

    assert "tests/test_compute_accounting.py" in makefile
    assert "make efficiency-check" in workflow


def test_public_compute_cli_derives_records_from_a_benchmark_artifact(tmp_path: Path):
    benchmark = _benchmark_record(tmp_path)
    benchmark_path = tmp_path / "benchmark.json"
    compute_path = tmp_path / "compute.json"
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/account_compute.py",
            "--benchmark",
            str(benchmark_path),
            "--output",
            str(compute_path),
            "--report",
            str(report_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    compute = json.loads(compute_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert compute["identity"]["benchmark_record_id"] == benchmark["record_id"]
    assert report["identity"]["operation_record_id"] == compute["operation_record_id"]
