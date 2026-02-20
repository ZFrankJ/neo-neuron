"""Summarize captured traces."""

import math
from typing import Any, Dict, Iterable, List


def _series_stats(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    total = sum(values)
    mean = total / len(values)
    return {"mean": mean, "min": min(values), "max": max(values)}


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def _abs_percentiles(values: Iterable[float]) -> Dict[str, float]:
    abs_values = sorted(abs(float(v)) for v in values)
    if not abs_values:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "p99_over_p50": 0.0,
        }
    p50 = _percentile(abs_values, 0.50)
    p99 = _percentile(abs_values, 0.99)
    return {
        "p50": p50,
        "p90": _percentile(abs_values, 0.90),
        "p95": _percentile(abs_values, 0.95),
        "p99": p99,
        "max": abs_values[-1],
        "p99_over_p50": p99 / max(p50, 1e-12),
    }


def summarize_neuron_records(records: Dict[int, Dict[str, object]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    state_all: List[float] = []
    output_all: List[float] = []
    hidden_all: List[float] = []
    cell_all: List[float] = []
    fx_all: List[float] = []
    gx_all: List[float] = []
    for layer_idx, layer_rec in records.items():
        layer_summary = {"neuron_indices": layer_rec.get("neuron_indices", [])}
        for key in ("embedding", "state", "output", "hidden_state", "cell_state", "f_x_raw", "g_x_raw"):
            if key not in layer_rec:
                continue
            stats = {}
            for nid, series in layer_rec[key].items():
                stats[nid] = _series_stats(series)
                if key == "state":
                    state_all.extend(series)
                elif key == "output":
                    output_all.extend(series)
                elif key == "hidden_state":
                    hidden_all.extend(series)
                elif key == "cell_state":
                    cell_all.extend(series)
                elif key == "f_x_raw":
                    fx_all.extend(series)
                elif key == "g_x_raw":
                    gx_all.extend(series)
            layer_summary[key] = stats
        if "gates" in layer_rec:
            gate_stats = {}
            for gate_name, gate_series in layer_rec["gates"].items():
                gate_stats[gate_name] = {nid: _series_stats(series) for nid, series in gate_series.items()}
            layer_summary["gates"] = gate_stats
        summary[str(layer_idx)] = layer_summary

    global_abs: Dict[str, Dict[str, float]] = {}
    if state_all:
        global_abs["state"] = _abs_percentiles(state_all)
    if output_all:
        global_abs["output"] = _abs_percentiles(output_all)
    if hidden_all:
        global_abs["hidden_state"] = _abs_percentiles(hidden_all)
    if cell_all:
        global_abs["cell_state"] = _abs_percentiles(cell_all)
    if fx_all:
        global_abs["f_x_raw"] = _abs_percentiles(fx_all)
    if gx_all:
        global_abs["g_x_raw"] = _abs_percentiles(gx_all)
    if global_abs:
        summary["_global_abs_percentiles"] = global_abs
    return summary
