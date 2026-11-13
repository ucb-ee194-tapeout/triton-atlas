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

Run all cases with:

```sh
make test T=test/from-jit
```

Every listed case is an end-to-end compiler target: Triton source must reach
Atlas virtual IR. Runtime execution and numerical comparison are separate
future tests.
