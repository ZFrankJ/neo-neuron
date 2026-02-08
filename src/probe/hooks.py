"""Hook helpers for probing models."""

import random
from typing import Dict, List, Tuple

import torch


def select_random_neurons(
    model,
    n_layers_to_watch: int = 3,
    neurons_per_layer: int = 3,
    seed: int = 0,
) -> Tuple[List[int], Dict[int, List[int]]]:
    if hasattr(model, "recurrent"):
        n_layers = int(model.recurrent.n_layers)
        d_model = int(model.d_model)
    elif hasattr(model, "lstm"):
        n_layers = int(model.lstm.num_layers)
        d_model = int(model.d_model)
    else:
        raise ValueError("Model does not expose a recurrent stack or LSTM.")

    rng = random.Random(seed)
    layer_indices = sorted(rng.sample(range(n_layers), k=min(n_layers_to_watch, n_layers)))
    selected_neurons: Dict[int, List[int]] = {}
    for li in layer_indices:
        count = min(neurons_per_layer, d_model)
        selected_neurons[li] = sorted(rng.sample(range(d_model), k=count))
    return layer_indices, selected_neurons


def attach_neo_hooks(model, layer_indices, selected_neurons, batch_index: int = 0):
    records: Dict[int, Dict[str, object]] = {}
    hooks = []

    for li in layer_indices:
        layer = model.recurrent.layers[li]
        neuron_ids = selected_neurons[li]
        records[li] = {
            "neuron_indices": neuron_ids,
            "embedding": {nid: [] for nid in neuron_ids},
            "state": {nid: [] for nid in neuron_ids},
            "output": {nid: [] for nid in neuron_ids},
            "f_x_raw": {nid: [] for nid in neuron_ids},
            "g_x_raw": {nid: [] for nid in neuron_ids},
        }

        def make_hook(layer_idx, neuron_ids_in_layer):
            def hook(_, __, outputs):
                out, state, aux = outputs
                b = min(batch_index, out.size(0) - 1)
                out_b = out[b].detach().cpu()
                state_b = state[b].detach().cpu()
                for nid in neuron_ids_in_layer:
                    records[layer_idx]["output"][nid].append(float(out_b[nid]))
                    records[layer_idx]["state"][nid].append(float(state_b[nid]))
                if isinstance(aux, dict):
                    f_x_raw = aux.get("f_x_raw")
                    g_x_raw = aux.get("g_x_raw")
                    if isinstance(f_x_raw, torch.Tensor) and isinstance(g_x_raw, torch.Tensor):
                        f_x_b = f_x_raw[b].detach().cpu()
                        g_x_b = g_x_raw[b].detach().cpu()
                        for nid in neuron_ids_in_layer:
                            records[layer_idx]["f_x_raw"][nid].append(float(f_x_b[nid]))
                            records[layer_idx]["g_x_raw"][nid].append(float(g_x_b[nid]))

            return hook

        hooks.append(layer.register_forward_hook(make_hook(li, neuron_ids)))

    return records, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
