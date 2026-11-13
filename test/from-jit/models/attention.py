"""Attention composition tests built from 32 by 32 Atlas tiles."""

import triton
import triton.language as tl


@triton.jit
def attention_then_mlp(Q, K, V, W0, W1, Out):
    """One attention block followed by a two-layer MLP."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    q = tl.load(Q + r * 32 + c)
    k = tl.load(K + r * 32 + c)
    v = tl.load(V + r * 32 + c)
    w0 = tl.load(W0 + r * 32 + c)
    w1 = tl.load(W1 + r * 32 + c)
    scores = tl.dot(q, tl.trans(k), input_precision='ieee').to(tl.bfloat16)
    row_max = tl.max(scores.to(tl.float32), 1).to(tl.bfloat16)
    exponent = tl.exp((scores - row_max[:, None]).to(tl.float32)).to(tl.bfloat16)
    row_sum = tl.sum(exponent.to(tl.float32), 1).to(tl.bfloat16)
    probability = (exponent * (1.0 / row_sum)[:, None]).to(tl.bfloat16)
    attended = tl.dot(probability.to(tl.float8e4nv, fp_downcast_rounding='rtne'), v,
                      input_precision='ieee').to(tl.bfloat16)
    hidden = tl.maximum(
        tl.dot(attended.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w0,
               input_precision='ieee'),
        0.0)
    output = tl.dot(hidden.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w1,
                    input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)


@triton.jit
def stacked_attention(Q, K0, V0, K1, V1, Out):
    """Two attention layers, with the first output becoming the second query."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    q0 = tl.load(Q + r * 32 + c)
    k0 = tl.load(K0 + r * 32 + c)
    v0 = tl.load(V0 + r * 32 + c)
    k1 = tl.load(K1 + r * 32 + c)
    v1 = tl.load(V1 + r * 32 + c)
    scores0 = tl.dot(q0, tl.trans(k0), input_precision='ieee').to(tl.bfloat16)
    max0 = tl.max(scores0.to(tl.float32), 1).to(tl.bfloat16)
    exp0 = tl.exp((scores0 - max0[:, None]).to(tl.float32)).to(tl.bfloat16)
    sum0 = tl.sum(exp0.to(tl.float32), 1).to(tl.bfloat16)
    prob0 = (exp0 * (1.0 / sum0)[:, None]).to(tl.bfloat16)
    q1 = tl.dot(prob0.to(tl.float8e4nv, fp_downcast_rounding='rtne'), v0,
                input_precision='ieee').to(tl.bfloat16)
    scores1 = tl.dot(q1.to(tl.float8e4nv, fp_downcast_rounding='rtne'),
                     tl.trans(k1), input_precision='ieee').to(tl.bfloat16)
    max1 = tl.max(scores1.to(tl.float32), 1).to(tl.bfloat16)
    exp1 = tl.exp((scores1 - max1[:, None]).to(tl.float32)).to(tl.bfloat16)
    sum1 = tl.sum(exp1.to(tl.float32), 1).to(tl.bfloat16)
    prob1 = (exp1 * (1.0 / sum1)[:, None]).to(tl.bfloat16)
    output = tl.dot(prob1.to(tl.float8e4nv, fp_downcast_rounding='rtne'), v1,
                    input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)


@triton.jit
def stacked_attention_then_mlp(Q, K0, V0, K1, V1, W0, W1, Out):
    """Two attention layers followed by a two-layer MLP."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    q0 = tl.load(Q + r * 32 + c)
    k0 = tl.load(K0 + r * 32 + c)
    v0 = tl.load(V0 + r * 32 + c)
    k1 = tl.load(K1 + r * 32 + c)
    v1 = tl.load(V1 + r * 32 + c)
    w0 = tl.load(W0 + r * 32 + c)
    w1 = tl.load(W1 + r * 32 + c)
    scores0 = tl.dot(q0, tl.trans(k0), input_precision='ieee').to(tl.bfloat16)
    max0 = tl.max(scores0.to(tl.float32), 1).to(tl.bfloat16)
    exp0 = tl.exp((scores0 - max0[:, None]).to(tl.float32)).to(tl.bfloat16)
    sum0 = tl.sum(exp0.to(tl.float32), 1).to(tl.bfloat16)
    prob0 = (exp0 * (1.0 / sum0)[:, None]).to(tl.bfloat16)
    q1 = tl.dot(prob0.to(tl.float8e4nv, fp_downcast_rounding='rtne'), v0,
                input_precision='ieee').to(tl.bfloat16)
    scores1 = tl.dot(q1.to(tl.float8e4nv, fp_downcast_rounding='rtne'),
                     tl.trans(k1), input_precision='ieee').to(tl.bfloat16)
    max1 = tl.max(scores1.to(tl.float32), 1).to(tl.bfloat16)
    exp1 = tl.exp((scores1 - max1[:, None]).to(tl.float32)).to(tl.bfloat16)
    sum1 = tl.sum(exp1.to(tl.float32), 1).to(tl.bfloat16)
    prob1 = (exp1 * (1.0 / sum1)[:, None]).to(tl.bfloat16)
    context = tl.dot(prob1.to(tl.float8e4nv, fp_downcast_rounding='rtne'), v1,
                     input_precision='ieee').to(tl.bfloat16)
    hidden = tl.maximum(
        tl.dot(context.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w0,
               input_precision='ieee'),
        0.0)
    output = tl.dot(hidden.to(tl.float8e4nv, fp_downcast_rounding='rtne'), w1,
                    input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)
