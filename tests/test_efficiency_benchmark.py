from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.runtime.efficiency import (
    SCHEMA_VERSION,
    BenchmarkObservation,
    BenchmarkSpec,
    benchmark_from_paths,
    run_benchmark,
    validate_benchmark_record,
    write_benchmark_record,
)
from src.runtime.backends import torch_backend


class _Clock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class _FakeAdapter:
    backend = "torch"
    device = "cpu"
    dtype = "float32"
    framework_version = "test-framework"
    hardware_identifier = "test-hardware"
    synchronization_policy = "fake.synchronize"
    telemetry_capabilities = {
        "process_rss": True,
        "backend_active_memory": False,
        "backend_peak_memory": False,
    }

    def __init__(self):
        self.events = []
        self._runs = 0

    def prepare(self, tokens, targets):
        self.events.append(("prepare", tuple(tokens.shape), tuple(targets.shape)))

    def reset_peak_memory(self):
        self.events.append("reset_peak")

    def synchronize(self):
        self.events.append("sync")

    def run(self, workload):
        self.events.append(("run", workload))
        self._runs += 1
        return BenchmarkObservation(
            output_shape=(4, 2, 17),
            finite=True,
            scalar=float(self._runs),
        )

    def memory_snapshot(self):
        self.events.append("memory")
        return {
            "process_rss_bytes": 1234,
            "backend_active_bytes": None,
            "backend_peak_bytes": None,
        }


def _metadata():
    return {
        "identity": {
            "git_commit": "abc123",
            "config_path": "/tmp/tiny.yaml",
            "config_sha256": "config-sha",
            "config_snapshot": {
                "model_name": "lstm",
                "vocab_size": 17,
                "use_checkpoint": False,
            },
            "checkpoint_path": "/tmp/tiny.pt",
            "checkpoint_sha256": "checkpoint-sha",
            "checkpoint_backend": "torch",
            "checkpoint_metadata": {"epoch": 1, "global_step": 2},
        },
        "model": {
            "name": "lstm",
            "profile_label": "tiny-test",
            "weight_provenance": "mapped_same_checkpoint",
            "trainable_parameters": 42,
            "parameter_breakdown": {"recurrent": 20, "other": 22},
            "activation_id": None,
            "recurrent_norm": "rmsnorm",
            "recurrent_norm_place": "all",
            "use_checkpoint": False,
        },
    }


def _spec(**overrides):
    values = {
        "workload": "sequence_eval",
        "timing_scope": "model_only",
        "batch_size": 2,
        "sequence_length": 4,
        "warmup_iterations": 1,
        "measured_iterations": 2,
        "repetition_id": "rep-1",
        "seed": 7,
        "dry_run": True,
    }
    values.update(overrides)
    return BenchmarkSpec(**values)


def test_shared_core_excludes_warmup_and_synchronizes_each_measured_region():
    adapter = _FakeAdapter()
    record = run_benchmark(
        adapter,
        _spec(),
        _metadata(),
        clock_ns=_Clock([100, 110, 200, 230]),
    )

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["raw_samples_ns"] == [10, 30]
    assert record["summary"]["median_ns"] == pytest.approx(20)
    assert record["summary"]["p10_ns"] == pytest.approx(12)
    assert record["summary"]["p90_ns"] == pytest.approx(28)
    assert record["summary"]["tokens_per_second"] == pytest.approx(4e8)
    assert record["workload"]["tokens_per_iteration"] == 8
    assert record["workload"]["data_handling_included"] is False
    assert record["workload"]["dry_run"] is True
    assert record["runtime"]["synchronization_policy"] == "fake.synchronize"
    assert record["telemetry_capabilities"]["backend_peak_memory"] is False
    assert record["memory"]["backend_peak_bytes"] is None
    assert record["output"] == {
        "shape": [4, 2, 17],
        "finite": True,
        "last_scalar": 3.0,
    }
    assert adapter.events == [
        ("prepare", (4, 2), (4, 2)),
        ("run", "sequence_eval"),
        "sync",
        "reset_peak",
        "sync",
        ("run", "sequence_eval"),
        "sync",
        "sync",
        ("run", "sequence_eval"),
        "sync",
        "memory",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"workload": "unknown"}, "workload"),
        ({"timing_scope": "combined"}, "timing_scope"),
        ({"workload": "train_step"}, "end_to_end_loop"),
        ({"workload": "sequence_eval", "timing_scope": "end_to_end_loop"}, "model_only"),
        ({"measured_iterations": 0}, "measured_iterations"),
        ({"warmup_iterations": 19, "measured_iterations": 100, "dry_run": False}, "20 warm-up"),
        ({"warmup_iterations": 20, "measured_iterations": 99, "dry_run": False}, "100 measured"),
    ],
)
def test_benchmark_spec_rejects_unsupported_or_underpowered_formal_runs(overrides, message):
    with pytest.raises(ValueError, match=message):
        _spec(**overrides)


def test_record_validation_rejects_capability_and_non_finite_mismatches():
    record = run_benchmark(
        _FakeAdapter(),
        _spec(measured_iterations=1),
        _metadata(),
        clock_ns=_Clock([10, 20]),
    )

    bad_capability = copy.deepcopy(record)
    bad_capability["telemetry_capabilities"]["backend_peak_memory"] = True
    with pytest.raises(ValueError, match="backend_peak_memory"):
        validate_benchmark_record(bad_capability)

    non_finite = copy.deepcopy(record)
    non_finite["output"]["finite"] = False
    with pytest.raises(ValueError, match="non-finite"):
        validate_benchmark_record(non_finite)


def test_authoritative_record_requires_explicit_replace(tmp_path: Path):
    record = run_benchmark(
        _FakeAdapter(),
        _spec(measured_iterations=1),
        _metadata(),
        clock_ns=_Clock([10, 20]),
    )
    output = tmp_path / "record.json"

    write_benchmark_record(output, record)
    with pytest.raises(FileExistsError, match="replace"):
        write_benchmark_record(output, record)
    write_benchmark_record(output, record, replace=True)

    assert json.loads(output.read_text(encoding="utf-8"))["record_id"] == record["record_id"]


def _tiny_cfg(model_name: str = "lstm"):
    cfg = {
        "model_name": "lstm",
        "vocab_size": 19,
        "d_model": 4,
        "d_embed": 4,
        "n_layers": 1,
        "dropout": 0.0,
        "tie_embeddings": True,
        "recurrent_norm": "rmsnorm",
        "recurrent_norm_place": "all",
        "rmsnorm_eps": 1e-5,
        "lstm_bias_mode": "single",
        "reference_backend": "mlx",
        "use_checkpoint": False,
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    if model_name == "neo":
        cfg.update(
            {
                "model_name": "neo",
                "cell_type": "cortical",
                "activation_id": "tanh",
                "weight_decay_policy": "table",
            }
        )
        cfg.pop("lstm_bias_mode")
    return cfg


def _write_tiny_fixture(tmp_path: Path, *, model_name: str = "lstm"):
    import yaml

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _tiny_cfg(model_name)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoint_path = tmp_path / "tiny.pt"
    model = torch_backend.build_model(cfg, model_name)
    torch_backend.save_checkpoint_entry(
        checkpoint_path,
        model,
        None,
        None,
        epoch=1,
        global_step=2,
        cfg=cfg,
    )
    return cfg, config_path, checkpoint_path


def _run_tiny_backend(
    tmp_path: Path,
    backend: str,
    device: str,
    *,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    workload: str = "sequence_eval",
    timing_scope: str = "model_only",
):
    if config_path is None or checkpoint_path is None:
        _, config_path, checkpoint_path = _write_tiny_fixture(tmp_path)
    output = tmp_path / f"{backend}-{workload}.json"
    record = benchmark_from_paths(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        backend_name=backend,
        device=device,
        workload=workload,
        timing_scope=timing_scope,
        batch_size=2,
        sequence_length=4,
        warmup_iterations=1,
        measured_iterations=2,
        repetition_id=f"{backend}-rep-1",
        profile_label="tiny-mapped-lstm",
        weight_provenance="mapped_same_checkpoint",
        output_path=output,
        seed=17,
        dry_run=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == record
    return record


def test_tiny_torch_cpu_integration_emits_valid_versioned_record(tmp_path: Path):
    record = _run_tiny_backend(tmp_path, "torch", "cpu")

    assert record["runtime"]["backend"] == "torch"
    assert record["runtime"]["device"] == "cpu"
    assert record["identity"]["checkpoint_backend"] == "torch"
    assert record["model"]["trainable_parameters"] > 0
    assert record["output"]["shape"] == [4, 2, 19]
    assert len(record["raw_samples_ns"]) == 2
    assert record["memory"]["process_rss_bytes"] > 0
    validate_benchmark_record(record)


@pytest.mark.parametrize(
    ("workload", "timing_scope"),
    [
        ("train_step", "end_to_end_loop"),
        ("sequence_eval", "model_only"),
        ("streaming_decode", "model_only"),
    ],
)
def test_tiny_torch_cpu_supports_each_workload(tmp_path: Path, workload, timing_scope):
    record = _run_tiny_backend(
        tmp_path,
        "torch",
        "cpu",
        workload=workload,
        timing_scope=timing_scope,
    )
    assert record["workload"]["name"] == workload
    assert record["workload"]["timing_scope"] == timing_scope
    assert record["output"]["finite"] is True


def test_checkpoint_metadata_mismatch_fails_before_execution(tmp_path: Path):
    import yaml

    cfg, config_path, checkpoint_path = _write_tiny_fixture(tmp_path)
    cfg["d_model"] = 5
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="Checkpoint metadata mismatch.*d_model"):
        benchmark_from_paths(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            backend_name="torch",
            device="cpu",
            workload="sequence_eval",
            timing_scope="model_only",
            batch_size=2,
            sequence_length=4,
            warmup_iterations=1,
            measured_iterations=1,
            repetition_id="mismatch",
            profile_label="tiny-mismatch",
            weight_provenance="mapped_same_checkpoint",
            output_path=tmp_path / "must-not-exist.json",
            seed=17,
            dry_run=True,
        )


def test_public_cli_writes_the_same_versioned_schema(tmp_path: Path):
    _, config_path, checkpoint_path = _write_tiny_fixture(tmp_path)
    output = tmp_path / "cli.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_efficiency.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--backend",
            "torch",
            "--device",
            "cpu",
            "--workload",
            "sequence_eval",
            "--timing-scope",
            "model_only",
            "--batch-size",
            "2",
            "--sequence-length",
            "4",
            "--warmup-iterations",
            "1",
            "--measured-iterations",
            "1",
            "--repetition-id",
            "cli-rep-1",
            "--profile-label",
            "tiny-cli",
            "--weight-provenance",
            "mapped_same_checkpoint",
            "--seed",
            "17",
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["workload"]["repetition_id"] == "cli-rep-1"
    assert record["record_id"] in completed.stdout


def test_tiny_mapped_torch_and_mlx_integrations_share_logical_schema(tmp_path: Path):
    mx = pytest.importorskip("mlx.core")
    original_device = mx.default_device()
    _, config_path, checkpoint_path = _write_tiny_fixture(tmp_path / "fixture")
    torch_record = _run_tiny_backend(
        tmp_path / "torch",
        "torch",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    mlx_record = _run_tiny_backend(
        tmp_path / "mlx",
        "mlx",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    assert mx.default_device() == original_device
    assert set(torch_record) == set(mlx_record)
    assert set(torch_record["runtime"]) == set(mlx_record["runtime"])
    assert set(torch_record["memory"]) == set(mlx_record["memory"])
    assert torch_record["model"]["trainable_parameters"] == mlx_record["model"]["trainable_parameters"]
    assert torch_record["model"]["parameter_breakdown"] == mlx_record["model"]["parameter_breakdown"]
    assert torch_record["workload"]["logical_shape"] == mlx_record["workload"]["logical_shape"]
    assert torch_record["output"]["shape"] == mlx_record["output"]["shape"]
    assert torch_record["output"]["last_scalar"] == pytest.approx(
        mlx_record["output"]["last_scalar"], rel=1e-5, abs=1e-6
    )
    assert torch_record["identity"]["checkpoint_sha256"] == mlx_record["identity"]["checkpoint_sha256"]


def test_tiny_mapped_neo_workload_matches_across_backends(tmp_path: Path):
    pytest.importorskip("mlx.core")
    _, config_path, checkpoint_path = _write_tiny_fixture(
        tmp_path / "fixture",
        model_name="neo",
    )
    torch_record = _run_tiny_backend(
        tmp_path / "torch",
        "torch",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    mlx_record = _run_tiny_backend(
        tmp_path / "mlx",
        "mlx",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    assert torch_record["model"]["name"] == mlx_record["model"]["name"] == "neo"
    assert torch_record["model"]["parameter_breakdown"] == mlx_record["model"]["parameter_breakdown"]
    assert torch_record["output"]["shape"] == mlx_record["output"]["shape"]
    assert torch_record["output"]["last_scalar"] == pytest.approx(
        mlx_record["output"]["last_scalar"], rel=1e-5, abs=1e-6
    )


@pytest.mark.parametrize(
    ("workload", "timing_scope"),
    [
        ("train_step", "end_to_end_loop"),
        ("sequence_eval", "model_only"),
        ("streaming_decode", "model_only"),
    ],
)
def test_tiny_mlx_supports_each_workload_when_available(tmp_path: Path, workload, timing_scope):
    pytest.importorskip("mlx.core")
    record = _run_tiny_backend(
        tmp_path,
        "mlx",
        "cpu",
        workload=workload,
        timing_scope=timing_scope,
    )
    assert record["workload"]["name"] == workload
    assert record["output"]["finite"] is True
