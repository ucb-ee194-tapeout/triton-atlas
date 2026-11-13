# Atlas model suite

These are end-to-end `@triton.jit` regression kernels, each restricted to one
32 by 32 tile. They deliberately increase composition complexity while keeping
the same small hardware unit.

| Case | What it covers |
| --- | --- |
| `fp8_two_layer` | Two MLP layers and one internal handoff |
| `fp8_three_layer` | Three MLP layers and repeated handoffs |
| `fp8_four_layer` | Four MLP layers; depth regression |
| `fp8_simple_attention` | One attention layer: score, softmax, value |
| `attention_then_mlp` | Attention followed by two MLP layers |
| `stacked_attention` | Two attention layers in sequence |
| `stacked_attention_then_mlp` | Two attention layers followed by two MLP layers |
| `attention_then_ten_layer_mlp` | One attention layer followed by ten MLP layers |
| `smolvla_transformer_core` | Q/K/V projections, attention, output projection, and a two-layer MLP |

Run all cases with:

```sh
make test T=test/from-jit
```

Every listed case is an end-to-end compiler target: Triton source must reach
Atlas virtual IR. Runtime execution and numerical comparison are separate
future tests.

## SmolVLA source basis

`smolvla_transformer_core` is a one-tile regression derived from the
[BAR Merlin SmolVLA workload](https://github.com/ucb-bar/Merlin-IREE/tree/main/models/smolVLA)
and its transformer-block sequence. It intentionally tests the supported
compute core only; it is not the full external SmolVLA checkpoint. RMSNorm,
residual connections, image/token preprocessing, and runtime weight loading
remain outside this compiler test until their lowerings exist.
