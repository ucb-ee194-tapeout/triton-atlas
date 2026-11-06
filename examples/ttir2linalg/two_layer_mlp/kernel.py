"""Two-layer FP8 MLP used as the first ttir2linalg/triton-atlas join case."""

import triton
import triton.language as tl


@triton.jit
def fp8_two_layer(X, W0, W1, Out):
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    w0 = tl.load(W0 + r * 32 + c)
    w1 = tl.load(W1 + r * 32 + c)
    hidden = tl.dot(x, w0, input_precision="ieee").to(tl.bfloat16)
    hidden = tl.maximum(hidden, 0.0).to(tl.bfloat16)
    packed = hidden.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    output = tl.dot(packed, w1, input_precision="ieee").to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)
