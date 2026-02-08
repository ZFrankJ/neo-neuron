"""Plotting utilities for probe traces."""

import os
from typing import Dict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "This module requires 'matplotlib'.\n"
        "Install via: pip install matplotlib"
    ) from exc


def _plot_two_axis(time_axis, left_series, right_series, left_label, right_label, title, out_path):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlabel("Time step")
    ax1.set_ylabel(left_label, color="tab:blue")
    ax1.plot(time_axis, left_series, color="tab:blue", linewidth=1.5, label=left_label)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(right_label, color="tab:red")
    ax2.plot(time_axis, right_series, color="tab:red", linewidth=1.5, linestyle="--", label=right_label)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_neo_neuron(time_axis, state, output, f_x_raw, g_x_raw, title, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.set_ylabel("Value")
    ax1.plot(time_axis, state, linewidth=1.5, label="state", color="tab:blue")
    ax1.plot(time_axis, output, linewidth=1.2, linestyle="--", label="output", color="tab:red")
    ax1.legend()
    ax1.set_title(title)

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Raw gate value")
    ax2.plot(time_axis, f_x_raw, linewidth=1.2, label="f_x_raw", color="tab:green")
    ax2.plot(time_axis, g_x_raw, linewidth=1.2, linestyle="--", label="g_x_raw", color="tab:purple")
    ax2.legend()

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_neo_records(records: Dict[int, Dict[str, object]], seq_len: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    time_axis = list(range(seq_len))
    for layer_idx, layer_rec in records.items():
        neuron_ids = layer_rec["neuron_indices"]
        fx_map = layer_rec.get("f_x_raw", {})
        gx_map = layer_rec.get("g_x_raw", {})
        for nid in neuron_ids:
            states = layer_rec["state"][nid]
            outputs = layer_rec["output"][nid]
            fx_raw = fx_map.get(nid, [])
            gx_raw = gx_map.get(nid, [])
            L = min(len(states), len(outputs), len(fx_raw), len(gx_raw), seq_len)
            if L == 0:
                L = min(len(states), len(outputs), seq_len)
                if L == 0:
                    continue
                fx_raw = [0.0] * L
                gx_raw = [0.0] * L
            out_path = os.path.join(out_dir, f"layer{layer_idx}_neuron{nid}.png")
            _plot_neo_neuron(
                time_axis[:L],
                states[:L],
                outputs[:L],
                fx_raw[:L],
                gx_raw[:L],
                f"Layer {layer_idx} — Neuron {nid}",
                out_path,
            )


def plot_lstm_records(records: Dict[int, Dict[str, object]], seq_len: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    time_axis = list(range(seq_len))
    for layer_idx, layer_rec in records.items():
        neuron_ids = layer_rec["neuron_indices"]
        for nid in neuron_ids:
            h_series = layer_rec["hidden_state"][nid]
            c_series = layer_rec["cell_state"][nid]
            L = min(len(h_series), len(c_series), seq_len)
            out_path = os.path.join(out_dir, f"layer{layer_idx}_neuron{nid}.png")
            _plot_two_axis(
                time_axis[:L],
                c_series[:L],
                h_series[:L],
                "Cell state",
                "Hidden state",
                f"LSTM Layer {layer_idx} — Neuron {nid}",
                out_path,
            )

            gates = layer_rec["gates"]
            gate_out = os.path.join(out_dir, f"layer{layer_idx}_neuron{nid}_gates.png")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_xlabel("Time step")
            ax.set_ylabel("Gate value")
            for gate_name in ("i", "f", "g", "o"):
                series = gates[gate_name][nid]
                ax.plot(time_axis[:L], series[:L], linewidth=1.2, label=gate_name)
            ax.legend()
            ax.set_title(f"LSTM Layer {layer_idx} — Neuron {nid} gates")
            fig.tight_layout()
            plt.savefig(gate_out, dpi=150)
            plt.close(fig)
