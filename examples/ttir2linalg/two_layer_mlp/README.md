# `ttir2linalg` two-layer MLP integration seed

These files are the first cross-repository integration case for implementing
the native Triton-to-Atlas pass in `triton-atlas`.

- `input.ttir` is the small, static 32x32 FP8 two-layer MLP fixture from
  `ttir2linalg/examples/two_layer_mlp_fp8.ttir`. It is a hand-authored TTIR
  fixture, **not** a TorchInductor capture.
- `kernel.py` is the Triton JIT source compiled directly by the native Atlas
  backend. `capture_with_atlas_backend.py` saves the backend's own TTIR and
  `atlas_virtual` stage output without creating tensors or launching a GPU
  kernel.
- `reference_triton_3_6.ttir` and its `.capture.json`/`.source.py` are a
  genuine offline Triton 3.6 capture from the latest `amy/triton-atlas-compiler`
  branch of `ttir2linalg`. The recorded TTIR and source hashes match. It is a
  semantic reference; `triton-atlas` itself is pinned to Triton 3.9, so the
  direct plugin integration should use a fresh capture from `kernel.py`.
- `expected_atlas_virtual.mlir` is the corresponding Atlas virtual program
  from the existing `ttir2linalg` mapper. It is a reference target, **not**
  output currently produced by the C++ plugin pass.

The target pattern is: load three FP8 tiles, transpose the two weight tiles,
MXU0 matmul, BF16 ReLU, BF16-to-FP8 pack, MXU1 matmul, then BF16 output. The
reference uses `atlas-fp8-mlp-v1`; this is an explicit Atlas precision policy,
not general FP32-equivalent lowering.

## Integration status

The latest `amy/triton-atlas-compiler` branch in `ttir2linalg` already has a
Python TTIR analyzer/emitter and tests for this MLP pattern. That implementation
is the semantic reference for the native plugin; the repos stay separate. The
native C++ pass now has a fail-closed lowering for this specific static MLP.
The lit test consumes a genuine Triton capture, checks that the pass emits the
expected Atlas ops, and runs the Atlas stream verifier.

After building the plugin on the supported Linux x86-64 setup, compile the JIT
kernel through the plugin (compile-only; no GPU is needed):

```bash
source env.sh
python examples/ttir2linalg/two_layer_mlp/capture_with_atlas_backend.py \
  --out-dir "$PWD/out/two-layer-mlp"
```

The script saves both stages, requires Atlas virtual ops, and verifies the
generated program. The reference target's Atlas IR can also be checked
independently with:

```bash
triton-atlas-opt test/ttir2linalg/two_layer_mlp_reference.mlir \
  --verify-atlas-virtual-stream
```

The implemented target contains two MXU matmuls, two XLU weight transposes,
one BF16 ReLU, one BF16-to-FP8 pack, and the input/output boundary ops. It
keeps the operation order, tile ABI, and explicit `atlas-fp8-mlp-v1` precision
policy from the reference. After this test passes on BWRC, repeat the same
contract with a captured attention kernel and connect per-kernel outputs to
the whole-model graph in model2mlir.
