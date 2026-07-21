from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.runtime.checkpoint_compat import load_checkpoint_payload
from src.runtime.efficiency import (
    SCHEMA_VERSION,
    BenchmarkObservation,
    BenchmarkSpec,
    benchmark_from_paths,
    hardware_identifier,
    resolve_benchmark_execution_config,
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

    def finalize_observation(self, pending):
        self.events.append("finalize")
        return pending

    def memory_snapshot(self):
        self.events.append("memory")
        return {
            "process_rss_bytes": 1234,
            "backend_active_bytes": None,
            "backend_peak_bytes": None,
        }


class _PeakFakeAdapter(_FakeAdapter):
    telemetry_capabilities = {
        "process_rss": True,
        "backend_active_memory": True,
        "backend_peak_memory": True,
    }

    def __init__(self):
        super().__init__()
        self._memory_values = iter(
            [
                {
                    "process_rss_bytes": 1001,
                    "backend_active_bytes": 11,
                    "backend_peak_bytes": 200,
                },
                {
                    "process_rss_bytes": 1002,
                    "backend_active_bytes": 12,
                    "backend_peak_bytes": 100,
                },
            ]
        )

    def memory_snapshot(self):
        self.events.append("memory")
        return next(self._memory_values)


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
            "effective_config_snapshot": {
                "model_name": "lstm",
                "vocab_size": 17,
                "use_checkpoint": False,
            },
            "execution_overrides": {},
            "metadata_inferences": {},
            "checkpoint_path": "/tmp/tiny.pt",
            "checkpoint_sha256": "checkpoint-sha",
            "checkpoint_backend": "torch",
            "checkpoint_metadata": {
                "epoch": 1,
                "global_step": 2,
                "config_snapshot": {},
            },
        },
        "model": {
            "name": "lstm",
            "profile_label": "tiny-test",
            "weight_provenance": "backend_native_checkpoint",
            "trainable_parameters": 42,
            "parameter_breakdown": {"recurrent": 20, "other": 22},
            "activation_id": None,
            "recurrent_norm": "rmsnorm",
            "recurrent_norm_place": "all",
            "use_checkpoint": False,
        },
        "evidence": {
            "status": "provisional",
            "provisional_reasons": ["dry_run"],
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
        "finalize",
        "reset_peak",
        "sync",
        ("run", "sequence_eval"),
        "sync",
        "memory",
        "finalize",
        "reset_peak",
        "sync",
        ("run", "sequence_eval"),
        "sync",
        "memory",
        "finalize",
    ]


def test_shared_core_keeps_peak_from_measured_work_before_output_validation():
    adapter = _PeakFakeAdapter()
    record = run_benchmark(
        adapter,
        _spec(warmup_iterations=0),
        _metadata(),
        clock_ns=_Clock([100, 110, 200, 230]),
    )

    assert record["memory"] == {
        "process_rss_bytes": 1002,
        "backend_active_bytes": 12,
        "backend_peak_bytes": 200,
    }


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

    false_provenance = copy.deepcopy(record)
    false_provenance["model"]["weight_provenance"] = "mapped_same_checkpoint"
    with pytest.raises(ValueError, match="weight_provenance"):
        validate_benchmark_record(false_provenance)


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
        "lstm_layer_dropout": 0.0,
        "reference_backend": "mlx",
        "use_checkpoint": False,
        "run_tag": "tiny-mapped-lstm",
        "cosine": False,
        "grad_clip": 1.0,
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


def _write_tiny_fixture(
    tmp_path: Path,
    *,
    model_name: str = "lstm",
    overrides: dict | None = None,
):
    import yaml

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _tiny_cfg(model_name)
    cfg.update(overrides or {})
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
        output_path=output,
        seed=17,
        dry_run=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == record
    return record


def test_tiny_torch_cpu_integration_emits_valid_versioned_record(tmp_path: Path):
    record = _run_tiny_backend(
        tmp_path,
        "torch",
        "cpu",
        workload="train_step",
        timing_scope="end_to_end_loop",
    )

    assert record["runtime"]["backend"] == "torch"
    assert record["runtime"]["device"] == "cpu"
    assert record["identity"]["checkpoint_backend"] == "torch"
    assert record["model"]["trainable_parameters"] > 0
    assert record["output"]["shape"] == [4, 2, 19]
    assert len(record["raw_samples_ns"]) == 2
    assert record["memory"]["process_rss_bytes"] > 0
    assert record["model"]["weight_provenance"] == "backend_native_checkpoint"
    assert record["evidence"] == {
        "status": "provisional",
        "provisional_reasons": ["dry_run"],
    }
    assert record["workload"]["semantics"] == {
        "input_policy": "deterministic_preallocated",
        "optimizer_state_policy": "fresh_then_warmed",
        "rng_policy": "backend_local_seeded_before_model_construction_and_execution",
        "scheduler_included": False,
        "state_policy": "reset_each_iteration",
        "tbptt_policy": "full_sequence",
    }
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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("d_model", 5),
        ("dropout", 0.25),
        ("lstm_layer_dropout", 0.25),
        ("reference_backend", "torch"),
        ("grad_clip", 0.5),
    ],
)
def test_checkpoint_metadata_mismatch_fails_before_execution(tmp_path: Path, key, value):
    import yaml

    cfg, config_path, checkpoint_path = _write_tiny_fixture(tmp_path)
    cfg[key] = value
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"Checkpoint metadata mismatch.*{key}"):
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
            output_path=tmp_path / "must-not-exist.json",
            seed=17,
            dry_run=True,
        )


def test_train_step_rejects_shorter_tbptt_contract(tmp_path: Path):
    _, config_path, checkpoint_path = _write_tiny_fixture(
        tmp_path,
        overrides={"tbptt_len": 2},
    )

    with pytest.raises(ValueError, match="tbptt_len"):
        benchmark_from_paths(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            backend_name="torch",
            device="cpu",
            workload="train_step",
            timing_scope="end_to_end_loop",
            batch_size=2,
            sequence_length=4,
            warmup_iterations=1,
            measured_iterations=1,
            repetition_id="tbptt-mismatch",
            profile_label="tiny-tbptt",
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
    assert record["model"]["weight_provenance"] == "backend_native_checkpoint"
    assert record["evidence"]["status"] == "provisional"
    assert record["record_id"] in completed.stdout


def test_hardware_identifier_records_platform_model(monkeypatch):
    monkeypatch.setattr("src.runtime.efficiency.platform.system", lambda: "Darwin")
    monkeypatch.setattr("src.runtime.efficiency.platform.machine", lambda: "arm64")

    class _Completed:
        def __init__(self, stdout):
            self.stdout = stdout

    values = {
        "machdep.cpu.brand_string": "Apple M4 Pro\n",
        "hw.model": "Mac16,7\n",
    }

    def fake_run(command, **kwargs):
        return _Completed(values[command[-1]])

    monkeypatch.setattr("src.runtime.efficiency.subprocess.run", fake_run)

    assert hardware_identifier() == "Darwin | arm64 | Apple M4 Pro | Mac16,7"


def test_readme_benchmark_example_uses_shell_safe_paths():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    benchmark_example = readme.split("## Efficiency benchmark harness", 1)[1].split("```", 2)[1]
    assert "<config.yaml>" not in benchmark_example
    assert "<checkpoint.pt>" not in benchmark_example
    assert "<record.json>" not in benchmark_example


def test_historical_mlx_neo_checkpoint_flag_becomes_recorded_execution_override():
    cfg = _tiny_cfg("neo") | {"backend": "mlx", "use_checkpoint": True}

    effective, overrides = resolve_benchmark_execution_config(
        cfg,
        model_name="neo",
        checkpoint_backend="mlx",
    )

    assert cfg["use_checkpoint"] is True
    assert effective["use_checkpoint"] is False
    assert overrides == {
        "use_checkpoint": {
            "training_value": True,
            "benchmark_value": False,
            "reason": "mlx_neo_training_flag_runtime_inert",
        }
    }


def test_non_mlx_checkpoint_flag_cannot_be_reinterpreted_for_benchmarking():
    cfg = _tiny_cfg("neo") | {"backend": "torch", "use_checkpoint": True}

    with pytest.raises(ValueError, match="use_checkpoint=false"):
        resolve_benchmark_execution_config(
            cfg,
            model_name="neo",
            checkpoint_backend="torch",
        )


def test_mlx_native_neo_checkpoint_with_historical_flag_is_benchmarkable(tmp_path: Path):
    mx = pytest.importorskip("mlx.core")
    import yaml
    from src.runtime.backends import mlx_backend

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/wt103/neo_20m.yaml").read_text(encoding="utf-8"))
    assert "reference_backend" not in cfg
    assert "rmsnorm_eps" not in cfg
    cfg.update(
        {
            "vocab_size": 19,
            "d_model": 4,
            "d_embed": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "block_size": 4,
            "batch_size": 2,
        }
    )
    config_path = tmp_path / "neo.yaml"
    checkpoint_path = tmp_path / "neo.pkl"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    original_device = mx.default_device()
    try:
        device = mlx_backend.get_runtime_device("cpu")
        mlx_backend.seed_all(23)
        model = mlx_backend.build_model(cfg, "neo")
        mlx_backend.save_checkpoint_entry(
            checkpoint_path,
            model,
            None,
            None,
            epoch=1,
            global_step=2,
            cfg=cfg,
        )
        record = benchmark_from_paths(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            backend_name="mlx",
            device=device,
            workload="sequence_eval",
            timing_scope="model_only",
            batch_size=2,
            sequence_length=4,
            warmup_iterations=20,
            measured_iterations=100,
            repetition_id="historical-neo",
            profile_label=cfg["run_tag"],
            output_path=tmp_path / "neo-record.json",
            seed=23,
            dry_run=False,
        )
    finally:
        mx.set_default_device(original_device)

    assert record["identity"]["config_snapshot"]["use_checkpoint"] is True
    assert "reference_backend" not in record["identity"]["config_snapshot"]
    assert "rmsnorm_eps" not in record["identity"]["config_snapshot"]
    assert "reference_backend" not in record["identity"]["checkpoint_metadata"]["config_snapshot"]
    assert "rmsnorm_eps" not in record["identity"]["checkpoint_metadata"]["config_snapshot"]
    assert record["identity"]["effective_config_snapshot"]["use_checkpoint"] is False
    assert record["identity"]["effective_config_snapshot"]["reference_backend"] == "mlx"
    assert record["identity"]["effective_config_snapshot"]["rmsnorm_eps"] == pytest.approx(1e-5)
    assert record["identity"]["execution_overrides"]["use_checkpoint"]["reason"] == (
        "mlx_neo_training_flag_runtime_inert"
    )
    assert record["identity"]["metadata_inferences"] == {
        "reference_backend": {
            "value": "mlx",
            "reason": "frozen_historical_mlx_neo_semantics",
            "config_sha256": record["identity"]["config_sha256"],
            "checkpoint_sha256": record["identity"]["checkpoint_sha256"],
        },
        "rmsnorm_eps": {
            "value": 1e-5,
            "reason": "frozen_historical_mlx_neo_semantics",
            "config_sha256": record["identity"]["config_sha256"],
            "checkpoint_sha256": record["identity"]["checkpoint_sha256"],
        },
    }
    assert record["model"]["use_checkpoint"] is False
    assert record["model"]["weight_provenance"] == "backend_native_checkpoint"
    assert record["evidence"] == {
        "status": "authoritative",
        "provisional_reasons": [],
    }

    tampered = copy.deepcopy(record)
    tampered["identity"]["metadata_inferences"]["rmsnorm_eps"]["checkpoint_sha256"] = (
        "not-the-checkpoint-hash"
    )
    with pytest.raises(ValueError, match="metadata_inferences"):
        validate_benchmark_record(tampered)


def test_missing_aligned_checkpoint_metadata_is_only_allowed_as_provisional_dry_run(
    tmp_path: Path,
):
    cfg, config_path, checkpoint_path = _write_tiny_fixture(tmp_path / "fixture")
    payload = load_checkpoint_payload(checkpoint_path)
    payload["cfg"].pop("lstm_bias_mode")
    payload["cfg"].pop("rmsnorm_eps")
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="Formal benchmark.*missing aligned metadata"):
        benchmark_from_paths(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            backend_name="torch",
            device="cpu",
            workload="sequence_eval",
            timing_scope="model_only",
            batch_size=2,
            sequence_length=4,
            warmup_iterations=20,
            measured_iterations=100,
            repetition_id="formal-missing-metadata",
            profile_label=cfg["run_tag"],
            output_path=tmp_path / "formal-must-not-exist.json",
            seed=17,
            dry_run=False,
        )

    record = benchmark_from_paths(
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
        repetition_id="provisional-missing-metadata",
        profile_label=cfg["run_tag"],
        output_path=tmp_path / "provisional.json",
        seed=17,
        dry_run=True,
    )
    assert record["evidence"]["status"] == "provisional"
    assert any("lstm_bias_mode" in reason for reason in record["evidence"]["provisional_reasons"])


def test_formal_record_rejects_profile_label_not_bound_to_config(tmp_path: Path):
    _, config_path, checkpoint_path = _write_tiny_fixture(tmp_path)

    with pytest.raises(ValueError, match="profile_label.*run_tag"):
        benchmark_from_paths(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            backend_name="torch",
            device="cpu",
            workload="sequence_eval",
            timing_scope="model_only",
            batch_size=2,
            sequence_length=4,
            warmup_iterations=20,
            measured_iterations=100,
            repetition_id="false-label",
            profile_label="not-the-config-profile",
            output_path=tmp_path / "must-not-exist.json",
            seed=17,
            dry_run=False,
        )


def test_complete_formal_record_is_authoritative(tmp_path: Path):
    cfg, config_path, checkpoint_path = _write_tiny_fixture(tmp_path)
    record = benchmark_from_paths(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        backend_name="torch",
        device="cpu",
        workload="sequence_eval",
        timing_scope="model_only",
        batch_size=2,
        sequence_length=4,
        warmup_iterations=20,
        measured_iterations=100,
        repetition_id="formal-complete",
        profile_label=cfg["run_tag"],
        output_path=tmp_path / "formal.json",
        seed=17,
        dry_run=False,
    )

    assert record["evidence"] == {
        "status": "authoritative",
        "provisional_reasons": [],
    }
    assert record["model"]["weight_provenance"] == "backend_native_checkpoint"


def test_torch_train_step_seed_is_backend_local_and_repeatable(tmp_path: Path):
    _, config_path, checkpoint_path = _write_tiny_fixture(
        tmp_path / "fixture",
        overrides={"dropout": 0.25},
    )
    first = _run_tiny_backend(
        tmp_path / "first",
        "torch",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        workload="train_step",
        timing_scope="end_to_end_loop",
    )
    second = _run_tiny_backend(
        tmp_path / "second",
        "torch",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        workload="train_step",
        timing_scope="end_to_end_loop",
    )

    assert first["output"]["last_scalar"] == second["output"]["last_scalar"]


def test_mlx_train_step_seed_is_backend_local_and_repeatable(tmp_path: Path):
    pytest.importorskip("mlx.core")
    _, config_path, checkpoint_path = _write_tiny_fixture(
        tmp_path / "fixture",
        overrides={"dropout": 0.25},
    )
    first = _run_tiny_backend(
        tmp_path / "first",
        "mlx",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        workload="train_step",
        timing_scope="end_to_end_loop",
    )
    second = _run_tiny_backend(
        tmp_path / "second",
        "mlx",
        "cpu",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        workload="train_step",
        timing_scope="end_to_end_loop",
    )

    assert first["output"]["last_scalar"] == second["output"]["last_scalar"]


def test_macos_ci_runs_efficiency_check_target():
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/tests.yml").read_text(encoding="utf-8")

    assert "efficiency-check:" in makefile
    assert "tests/test_efficiency_benchmark.py" in makefile
    assert "make efficiency-check" in workflow


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
