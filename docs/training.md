# Training

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
