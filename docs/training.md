# Training

## Config labels and activation provenance

The active WT103 Neo templates use `activation_id: tanh`, and their run tags
include the activation (`wt103_neo_<size>_tanh`). The size labels are rounded
total-parameter labels; always report the exact parameter count printed by
`scripts/train.py` with paper-facing results. The current Torch counts are
20,356,863, 30,351,943, and 50,342,103 parameters for the 20m, 30m, and 50m
Neo templates respectively.

Historical checkpoints and run snapshots are authoritative for their own
activation metadata. In particular, old `id4` and `id5` runs must keep those
labels and must not be described as tanh runs.

The WT2 filenames are retained only as compatibility paths for existing
scripts and records. Their numeric suffixes are legacy names, not trustworthy
parameter counts:

| Compatibility path | Exact Torch parameter count | Reporting label |
| --- | ---: | --- |
| `configs/wt2/neo_6m.yaml` | 9,708,894 | WT2 Neo small (legacy `6m`, `id4`) |
| `configs/wt2/lstm_6m.yaml` | 9,636,561 | WT2 LSTM small (legacy `6m`) |
| `configs/wt2/lstm_25m.yaml` | 28,589,265 | WT2 LSTM large (legacy `25m`) |

Do not derive a parameter count from these WT2 filenames. Future PR bodies and
result tables should use the reporting label, exact count, activation ID for
Neo, and evaluation regime.

## LSTM baseline taxonomy

Two LSTM control labels are available for future result reporting:

- `RMSNorm-LSTM matched control` preserves existing config behavior: zero gate
  biases, Xavier-uniform input and recurrent matrices, and `dropout` reused as
  inter-layer and output dropout.
- `standard-init RMSNorm-LSTM` is an opt-in PyTorch control using
  `forget_bias_init: 1.0`, `recurrent_init: orthogonal`, and an explicit
  `lstm_layer_dropout` policy. Gate-wise orthogonal initialization is applied to
  each hidden-to-hidden gate matrix. The forget value is written to the input
  forget bias while the hidden forget bias remains zero, so the effective
  forget-gate bias is exactly the configured value.

When `lstm_layer_dropout` is absent, it defaults to `dropout` for compatibility.
Setting it to `0.0` removes only LSTM inter-layer dropout; `dropout` still applies
before the output projection. Supported `recurrent_init` values are
`xavier_uniform` and `orthogonal`.

Do not relabel historical runs as standard-init runs. The new controls affect
parameter initialization and therefore only apply when starting a new PyTorch
LSTM run; loaded checkpoint weights remain authoritative. MLX LSTM semantics are
unchanged by this control path.

## Recurrent evaluation regimes

Evaluation reports an explicit `eval_regime` for Neo and LSTM models:

- `block_reset` initializes recurrent state for every non-overlapping block.
  This is the compatibility default when `eval_regime` is absent, so existing
  configs and checkpoints keep their historical interpretation.
- `streaming` splits the token stream into contiguous batch lanes and carries
  each lane's recurrent state across successive blocks. State never crosses
  between lanes.

Set `eval_regime: streaming` in a config or override it for a standalone run
with `scripts/eval.py --eval-regime streaming`. The CLI prints the selected
regime before perplexity. Future result tables must state the regime; do not
reinterpret historical rows that did not opt into streaming evaluation.
