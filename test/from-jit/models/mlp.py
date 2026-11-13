"""MLP depth tests built from 32 by 32 Atlas tiles."""

import triton
import triton.language as tl


@triton.jit
def fp8_four_layer(X, W0, W1, W2, W3, Out):
    """Four layers: the deepest standalone MLP regression target."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    w0 = tl.load(W0 + r * 32 + c)
    w1 = tl.load(W1 + r * 32 + c)
    w2 = tl.load(W2 + r * 32 + c)
    w3 = tl.load(W3 + r * 32 + c)
    h0 = tl.maximum(tl.dot(x, w0, input_precision='ieee'), 0.0)
    h1 = tl.maximum(tl.dot(h0.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w1,
                           input_precision='ieee'), 0.0)
    h2 = tl.maximum(tl.dot(h1.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w2,
                           input_precision='ieee'), 0.0)
    out = tl.dot(h2.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w3,
                 input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, out)
