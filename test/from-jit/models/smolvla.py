"""A 32 by 32 SmolVLA transformer-core compiler regression.

This is derived from the BAR Merlin SmolVLA block shape, not a claim that the
full external checkpoint runs in this test.  It keeps the supported core:
Q/K/V projections, attention, output projection, and a two-layer MLP.
"""

import triton
import triton.language as tl


@triton.jit
def smolvla_transformer_core(X, WQ, WK, WV, WO, WUP, WDOWN, Out):
    """One single-tile transformer core based on SmolVLA's block structure."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    wq = tl.load(WQ + r * 32 + c)
    wk = tl.load(WK + r * 32 + c)
    wv = tl.load(WV + r * 32 + c)
    wo = tl.load(WO + r * 32 + c)
    wup = tl.load(WUP + r * 32 + c)
    wdown = tl.load(WDOWN + r * 32 + c)

    q = tl.dot(x, wq, input_precision='ieee').to(tl.bfloat16)
    k = tl.dot(x, wk, input_precision='ieee').to(tl.bfloat16)
    v = tl.dot(x, wv, input_precision='ieee').to(tl.bfloat16)
    scores = tl.dot(q.to(tl.float8e4nv, fp_downcast_rounding='rtne'),
                    tl.trans(k.to(tl.float8e4nv, fp_downcast_rounding='rtne')),
                    input_precision='ieee').to(tl.bfloat16)

    row_max = tl.max(scores.to(tl.float32), 1).to(tl.bfloat16)
    exponent = tl.exp((scores - row_max[:, None]).to(tl.float32)).to(tl.bfloat16)
    row_sum = tl.sum(exponent.to(tl.float32), 1).to(tl.bfloat16)
    probability = (exponent * (1.0 / row_sum)[:, None]).to(tl.bfloat16)
    context = tl.dot(probability.to(tl.float8e4nv, fp_downcast_rounding='rtne'),
                     v.to(tl.float8e4nv, fp_downcast_rounding='rtne'),
                     input_precision='ieee').to(tl.bfloat16)
    projected = tl.dot(context.to(tl.float8e4nv, fp_downcast_rounding='rtne'), wo,
                       input_precision='ieee').to(tl.bfloat16)
    hidden = tl.maximum(
        tl.dot(projected.to(tl.float8e4nv, fp_downcast_rounding='rtne'), wup,
               input_precision='ieee'), 0.0)
    output = tl.dot(hidden.to(tl.float8e4nv, fp_downcast_rounding='rtne'), wdown,
                    input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)
