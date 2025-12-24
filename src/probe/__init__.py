"""Probe utilities."""

from .hooks import select_random_neurons, attach_neo_hooks, remove_hooks
from .capture import capture_traces, capture_neo_traces, capture_lstm_traces
from .summarize import summarize_neuron_records

try:
    from .viz import plot_neo_records, plot_lstm_records
except Exception:  # pragma: no cover - optional dependency
    plot_neo_records = None
    plot_lstm_records = None

__all__ = [
    "select_random_neurons",
    "attach_neo_hooks",
    "remove_hooks",
    "capture_traces",
    "capture_neo_traces",
    "capture_lstm_traces",
    "summarize_neuron_records",
    "plot_neo_records",
    "plot_lstm_records",
]
