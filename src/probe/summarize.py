"""Summarize captured traces."""

from typing import Dict


def _series_stats(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    total = sum(values)
    mean = total / len(values)
    return {"mean": mean, "min": min(values), "max": max(values)}


def summarize_neuron_records(records: Dict[int, Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    summary: Dict[int, Dict[str, object]] = {}
    for layer_idx, layer_rec in records.items():
        layer_summary = {"neuron_indices": layer_rec.get("neuron_indices", [])}
        for key in ("state", "output", "hidden_state", "cell_state"):
            if key not in layer_rec:
                continue
            stats = {}
            for nid, series in layer_rec[key].items():
                stats[nid] = _series_stats(series)
            layer_summary[key] = stats
        if "gates" in layer_rec:
            gate_stats = {}
            for gate_name, gate_series in layer_rec["gates"].items():
                gate_stats[gate_name] = {nid: _series_stats(series) for nid, series in gate_series.items()}
            layer_summary["gates"] = gate_stats
        summary[layer_idx] = layer_summary
    return summary
