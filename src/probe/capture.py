"""Capture activation traces for probing."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .hooks import attach_neo_hooks, remove_hooks


def capture_neo_traces(
    model,
    idx: torch.Tensor,
    layer_indices: List[int],
    selected_neurons: Dict[int, List[int]],
    batch_index: int = 0,
) -> Dict[int, Dict[str, object]]:
    model.eval()
    records, hooks = attach_neo_hooks(model, layer_indices, selected_neurons, batch_index=batch_index)
    with torch.no_grad():
        x = model.emb(idx)
        if not isinstance(model.in_proj, torch.nn.Identity):
            x = model.in_proj(x)
        b = min(batch_index, x.size(1) - 1)
        x_b = x[:, b, :].detach().cpu()
        for li in layer_indices:
            emb_series = records[li]["embedding"]
            for nid in selected_neurons[li]:
                if nid < x_b.size(1):
                    emb_series[nid] = [float(v) for v in x_b[:, nid]]
                else:
                    emb_series[nid] = []
    with torch.no_grad():
        state = model.init_state(idx.size(1), idx.device) if hasattr(model, "init_state") else None
        model(idx, state)
    remove_hooks(hooks)
    return records


def capture_lstm_traces(
    model,
    idx: torch.Tensor,
    layer_indices: List[int],
    selected_neurons: Dict[int, List[int]],
    batch_index: int = 0,
) -> Dict[int, Dict[str, object]]:
    model.eval()
    device = idx.device
    T, B = idx.shape

    x = model.emb(idx)
    if not isinstance(model.in_proj, torch.nn.Identity):
        x = model.in_proj(x)

    n_layers = model.lstm.num_layers
    d_model = model.d_model
    h = torch.zeros(n_layers, B, d_model, device=device, dtype=x.dtype)
    c = torch.zeros(n_layers, B, d_model, device=device, dtype=x.dtype)

    records: Dict[int, Dict[str, object]] = {}
    for li in layer_indices:
        neuron_ids = selected_neurons[li]
        records[li] = {
            "neuron_indices": neuron_ids,
            "embedding": {nid: [] for nid in neuron_ids},
            "cell_state": {nid: [] for nid in neuron_ids},
            "hidden_state": {nid: [] for nid in neuron_ids},
            "gates": {
                "i": {nid: [] for nid in neuron_ids},
                "f": {nid: [] for nid in neuron_ids},
                "g": {nid: [] for nid in neuron_ids},
                "o": {nid: [] for nid in neuron_ids},
            },
        }

    with torch.no_grad():
        b = min(batch_index, B - 1)
        x_b = x[:, b, :].detach().cpu()
        for li in layer_indices:
            for nid in records[li]["neuron_indices"]:
                if nid < x_b.size(1):
                    records[li]["embedding"][nid] = [float(v) for v in x_b[:, nid]]

        for t in range(T):
            layer_input = x[t]
            for li in range(n_layers):
                h_prev = h[li]
                c_prev = c[li]
                weight_ih = getattr(model.lstm, f"weight_ih_l{li}")
                weight_hh = getattr(model.lstm, f"weight_hh_l{li}")
                bias_ih = getattr(model.lstm, f"bias_ih_l{li}")
                bias_hh = getattr(model.lstm, f"bias_hh_l{li}")
                pre_norms = getattr(model.lstm, "pre_norms", None)
                layer_input_norm = pre_norms[li](layer_input) if pre_norms is not None else layer_input
                gates = F.linear(layer_input_norm, weight_ih, bias_ih) + F.linear(h_prev, weight_hh, bias_hh)
                i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                g_gate = torch.tanh(g_gate)
                o_gate = torch.sigmoid(o_gate)

                c_new = f_gate * c_prev + i_gate * g_gate
                h_new = o_gate * torch.tanh(c_new)

                if li in records:
                    b = min(batch_index, B - 1)
                    h_b = h_new[b].detach().cpu()
                    c_b = c_new[b].detach().cpu()
                    i_b = i_gate[b].detach().cpu()
                    f_b = f_gate[b].detach().cpu()
                    g_b = g_gate[b].detach().cpu()
                    o_b = o_gate[b].detach().cpu()
                    for nid in records[li]["neuron_indices"]:
                        records[li]["hidden_state"][nid].append(float(h_b[nid]))
                        records[li]["cell_state"][nid].append(float(c_b[nid]))
                        records[li]["gates"]["i"][nid].append(float(i_b[nid]))
                        records[li]["gates"]["f"][nid].append(float(f_b[nid]))
                        records[li]["gates"]["g"][nid].append(float(g_b[nid]))
                        records[li]["gates"]["o"][nid].append(float(o_b[nid]))

                h[li] = h_new
                c[li] = c_new
                layer_input = h_new

    return records


def capture_traces(
    model,
    idx: torch.Tensor,
    layer_indices: List[int],
    selected_neurons: Dict[int, List[int]],
    batch_index: int = 0,
) -> Dict[int, Dict[str, object]]:
    if hasattr(model, "recurrent"):
        return capture_neo_traces(model, idx, layer_indices, selected_neurons, batch_index=batch_index)
    if hasattr(model, "lstm"):
        return capture_lstm_traces(model, idx, layer_indices, selected_neurons, batch_index=batch_index)
    raise ValueError("Model type not supported for probing.")
