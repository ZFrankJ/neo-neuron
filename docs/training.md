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

Backend-native historical LSTM profiles are distinct and must retain their
provenance:

- `native-MLX RMSNorm-LSTM` uses MLX-native uniform recurrent initialization,
  a single trainable gate-bias vector with native random initialization, and
  `dropout` as both inter-layer and output dropout.
- `legacy Torch RMSNorm-LSTM` uses Xavier-uniform input and recurrent matrices,
  two trainable zero-initialized bias vectors, and `dropout` as both inter-layer
  and output dropout.
- `standard-init Torch RMSNorm-LSTM` is an opt-in PyTorch control using
  `forget_bias_init: 1.0`, `recurrent_init: orthogonal`, and an explicit
  `lstm_layer_dropout` policy. Gate-wise orthogonal initialization is applied to
  each hidden-to-hidden gate matrix. The forget value is written to the input
  forget bias while the hidden forget bias remains zero, so the effective
  forget-gate bias is exactly the configured value.

When `lstm_layer_dropout` is absent, it defaults to `dropout` for compatibility.
Setting it to `0.0` removes only LSTM inter-layer dropout; `dropout` still applies
before the output projection. Supported `recurrent_init` values are
`xavier_uniform` and `orthogonal`.

`lstm_bias_mode` makes effective-bias training explicit. When it is absent,
Torch keeps historical `split` behavior and MLX keeps native `single` behavior.
Torch `single` mode retains `bias_hh` in checkpoint state but freezes it, so
only `bias_ih` receives optimizer updates and the trainable parameter count
matches MLX. MLX accepts explicit `single` and rejects `split`; it does not
change its native parameterization. Historical split-bias Torch checkpoints
remain evaluable in single mode because both saved bias tensors are retained in
the forward sum. Torch-to-MLX conversion sums them and warns that cross-backend
optimizer resume is not equivalent.

`rmsnorm_eps` makes the LSTM RMSNorm numerical contract explicit. Aligned
Torch/MLX LSTM profiles set `rmsnorm_eps: 1e-5`. A Torch config that omits the
field retains the historical PyTorch dtype-derived RMSNorm epsilon; it is not
silently promoted to the aligned profile. MLX preserves its native `1e-5`
behavior when the field is absent, accepts an explicit `1e-5`, and rejects any
other explicit LSTM value rather than ignoring it. Aligned LSTM checkpoints
record and validate `lstm_bias_mode`, `recurrent_norm`,
`recurrent_norm_place`, and `rmsnorm_eps`; checkpoints missing those fields are
loadable as legacy/provisional state with a warning.

The deterministic LSTM training-parity profile additionally requires
`reference_backend: mlx`, `lstm_bias_mode: single`, `dropout: 0.0`, fixed
batches, and `use_checkpoint: false`. Its test envelope is `1e-6` for loss,
`1e-5` for mapped gradients, one optimizer update, and the 12-step fixed-batch
trajectory; recurrent-state comparisons use `rtol=1e-5` and `atol=1e-6`.
Torch and MLX model initialization is reset to the fixed seed `20260714` before
each test. These thresholds were set above the measured
local maxima of `7.16e-7` loss drift, `5.93e-7` gradient-norm drift,
`2.99e-8` parameter drift, and `1.20e-7` recurrent-state drift. Exact replay of
random batches or dropout masks remains outside the contract.

Optimizer state is backend-native. Same-backend Torch and MLX checkpoint resume
is covered exactly. Cross-backend loads convert model weights only; when an
optimizer and saved optimizer state are both present, the loader warns that
cross-backend optimizer resume is unsupported and leaves the destination
optimizer state untouched.

Do not relabel historical runs as standard-init runs. The new controls affect
parameter initialization and therefore only apply when starting a new PyTorch
LSTM run; loaded checkpoint weights remain authoritative. A future matched
profile must state `lstm_bias_mode: single` explicitly. MLX LSTM runtime
semantics remain unchanged; explicit epsilon support only validates and records
the existing native value.

With `reference_backend: mlx` and table-based weight decay, Torch LSTM
parameters are grouped by role despite backend-specific names: `lstm.*`
recurrent matrices use `recurrent_weight_decay`, embeddings and projections use
their configured decay values, and biases plus recurrent norms use zero decay.
MLX keeps its native `lstm_layers.*` naming and unchanged runtime semantics.

## Transformer baseline taxonomy

Transformer configs without `transformer_variant` are the historical
`lightweight Transformer internal control`. They retain the old initialization
and checkpoint contract and are not a GPT-2 comparison baseline.

`transformer_variant: gpt2` selects the `GPT-2-style Transformer control`:
pre-norm blocks, learned absolute positions, GELU, optimized causal attention,
0.02 normal initialization, and `1 / sqrt(2 * n_layers)` scaling for attention
and MLP residual-projection initialization. Torch and MLX use their respective
optimized attention primitives and share checkpoint-compatible parameter
shapes. Paper-facing tables must state which Transformer variant was used.

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
