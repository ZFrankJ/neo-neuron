"""Contract tests for shared backend parity audit reports."""

from __future__ import annotations

import numpy as np
import pytest

from src.runtime.parity_audit import (
    BackendParityAuditReport,
    collect_array_metrics,
    device_availability,
    make_backend_parity_report,
)


def test_cpu_self_parity_audit_report_is_exact_and_structured():
    logits = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    state = np.array([[[0.01, -0.02], [0.03, 0.04]]], dtype=np.float32)
    gradients = {
        "emb.weight": np.array([[0.2, -0.1], [0.0, 0.3]], dtype=np.float32),
        "out.weight": np.array([[0.05, -0.04]], dtype=np.float32),
    }
    updates = {
        "emb.weight": np.array([[0.001, -0.002], [0.0, 0.003]], dtype=np.float32),
        "out.weight": np.array([[0.004, -0.005]], dtype=np.float32),
    }

    report = make_backend_parity_report(
        backend_pair=("torch", "torch"),
        device_pair=("cpu", "cpu"),
        seed=1234,
        model_shape={"vocab_size": 8, "d_model": 2, "n_layers": 1},
        use_checkpoint=False,
        loss_pair=(1.25, 1.25),
        logits_pair=(logits, logits.copy()),
        state_pair=(state, state.copy()),
        gradient_pair=(gradients, {name: value.copy() for name, value in gradients.items()}),
        update_pair=(updates, {name: value.copy() for name, value in updates.items()}),
        memory_samples=[{"device": "cpu", "allocated_bytes": 0}],
    )

    assert isinstance(report, BackendParityAuditReport)
    assert report.backend_pair == ("torch", "torch")
    assert report.device_pair == ("cpu", "cpu")
    assert report.seed == 1234
    assert report.model_shape == {"vocab_size": 8, "d_model": 2, "n_layers": 1}
    assert report.use_checkpoint is False
    assert report.loss_diff == pytest.approx(0.0)
    assert report.logits_max_diff == pytest.approx(0.0)
    assert report.state_max_diff == pytest.approx(0.0)
    assert report.gradient_max_diff == pytest.approx(0.0)
    assert report.update_max_diff == pytest.approx(0.0)
    assert report.grad_norm == pytest.approx(np.sqrt(0.2**2 + 0.1**2 + 0.3**2 + 0.05**2 + 0.04**2))
    assert report.nan_count == 0
    assert report.inf_count == 0
    assert report.memory_samples == [{"device": "cpu", "allocated_bytes": 0}]

    serialized = report.to_dict()
    assert serialized["backend_pair"] == ["torch", "torch"]
    assert serialized["device_pair"] == ["cpu", "cpu"]
    assert serialized["memory_samples"] == [{"device": "cpu", "allocated_bytes": 0}]


def test_collect_array_metrics_counts_non_finite_values():
    got = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    expected = np.array([1.5, 0.0, 3.0, -4.0], dtype=np.float32)

    metrics = collect_array_metrics(got, expected)

    assert metrics.max_abs_diff == pytest.approx(0.5)
    assert metrics.nan_count == 1
    assert metrics.inf_count == 2


def test_optional_torch_hardware_availability_reports_skip_reasons():
    cuda = device_availability("cuda")
    mps = device_availability("mps")

    assert cuda.device == "cuda"
    assert isinstance(cuda.available, bool)
    if not cuda.available:
        assert cuda.skip_reason

    assert mps.device == "mps"
    assert isinstance(mps.available, bool)
    if not mps.available:
        assert mps.skip_reason
