"""The @triton.jit kernels test_from_jit.py compiles. Each says where it comes from."""
import triton
import triton.language as tl


# The MLP fixture's kernel (test/fixtures/mlp.ttir.mlir): relu(x @ w.T + b) on one 32x32 tile.
@triton.jit
def mlp(X, W, B, Y):
    r = tl.arange(0, 32)
    off = r[:, None] * 32 + r[None, :]
    x = tl.load(X + off)
    w = tl.load(W + off)
    b = tl.load(B + r).to(tl.float32)
    acc = tl.dot(x, tl.trans(w)) + b[None, :]
    tl.store(Y + off, tl.maximum(acc, 0.0).to(tl.bfloat16))


# The matmul_relu fixture's kernel (test/fixtures/matmul_relu.ttir.mlir): relu(a @ b), f32 output.
@triton.jit
def matmul_relu(A, B, C):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    a = tl.load(A + rows[:, None] * 32 + cols[None, :])
    b = tl.load(B + rows[:, None] * 32 + cols[None, :])
    result = tl.maximum(tl.dot(a, b, tl.full((32, 32), 0.0, tl.float32)), 0.0)
    tl.store(C + rows[:, None] * 32 + cols[None, :], result)


# The mm32 fixture's kernel (test/fixtures/mm32.ttir.mlir): a K loop of 32x32 tiles, K = 128.
@triton.jit
def mm32(A, Bm, C, K: tl.constexpr):
    r = tl.arange(0, 32)
    a = A + r[:, None] * K + r[None, :]
    b = Bm + r[:, None] * 32 + r[None, :]
    acc = tl.zeros((32, 32), tl.float32)
    for _ in range(0, K // 32):
        acc = tl.dot(tl.load(a), tl.load(b), acc)
        a += 32
        b += 32 * 32
    tl.store(C + r[:, None] * 32 + r[None, :], acc.to(tl.bfloat16))


# Ours: a K loop whose accumulator starts from a bias, so pass 2 moves the bias after the whole chain.
@triton.jit
def bias_seeded_k_loop(A, Bm, Bias, C, K: tl.constexpr):
    r = tl.arange(0, 32)
    a = A + r[:, None] * K + r[None, :]
    b = Bm + r[:, None] * 32 + r[None, :]
    acc = tl.broadcast_to(tl.load(Bias + r).to(tl.float32)[None, :], (32, 32))
    for _ in range(0, K // 32):
        acc = tl.dot(tl.load(a), tl.load(b), acc)
        a += 32
        b += 32 * 32
    tl.store(C + r[:, None] * 32 + r[None, :], acc.to(tl.bfloat16))


# Ours: a K loop Triton unrolls itself (loop_unroll_factor=2), so its chain reaches pass 2 with no loop.
@triton.jit
def unrolled_by_triton(A, Bm, C, K: tl.constexpr):
    r = tl.arange(0, 32)
    a = A + r[:, None] * K + r[None, :]
    b = Bm + r[:, None] * 32 + r[None, :]
    acc = tl.zeros((32, 32), tl.float32)
    for _ in tl.range(0, K // 32, loop_unroll_factor=2):
        acc = tl.dot(tl.load(a), tl.load(b), acc)
        a += 32
        b += 32 * 32
    tl.store(C + r[:, None] * 32 + r[None, :], acc.to(tl.bfloat16))


# Ours: dot(x, w) + dot(w, x), which Triton's combine folds into one dot accumulating onto the other.
@triton.jit
def dot_plus_dot(A, Bm, C):
    r = tl.arange(0, 32)
    off = r[:, None] * 32 + r[None, :]
    x = tl.load(A + off)
    w = tl.load(Bm + off)
    tl.store(C + off, (tl.dot(x, w) + tl.dot(w, x)).to(tl.bfloat16))


# Ours: a row max, then a row sum of the shifted rows.
@triton.jit
def row_reduce(X, Y):
    r = tl.arange(0, 32)
    x = tl.load(X + r[:, None] * 32 + r[None, :]).to(tl.float32)
    m = tl.max(x, axis=1)
    s = tl.sum(x - m[:, None], axis=1)
    tl.store(Y + r, s.to(tl.bfloat16))


# From the team's ttir2linalg repo, kernels/atlas_jit_kernels.py lines 32-43 @ 3db3b518: two FP8 layers, packing
# the hidden layer back to FP8 in the kernel.
@triton.jit
def fp8_two_layer(X, W0, W1, Out):
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    w0 = tl.load(W0 + r * 32 + c)
    w1 = tl.load(W1 + r * 32 + c)
    hidden = tl.dot(x, w0, input_precision='ieee').to(tl.bfloat16)
    hidden = tl.maximum(hidden, 0.0).to(tl.bfloat16)
    packed = hidden.to(tl.float8e4nv, fp_downcast_rounding='rtne')
    output = tl.dot(packed, w1, input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)


# From the team's ttir2linalg repo, kernels/atlas_jit_kernels.py lines 74-93 @ 3db3b518: static BF16 attention on
# one tile.
@triton.jit
def bf16_simple_attention(Q, K, V, Out):
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    q = tl.load(Q + r * 32 + c)
    k = tl.load(K + r * 32 + c)
    v = tl.load(V + r * 32 + c)
    scores = (tl.dot(q, tl.trans(k), input_precision='ieee') * 0.1767766952966369).to(tl.bfloat16)
    maximum = tl.max(scores.to(tl.float32), 1).to(tl.bfloat16)
    shifted = (scores - maximum[:, None]).to(tl.bfloat16)
    exponential = tl.exp(shifted.to(tl.float32)).to(tl.bfloat16)
    denominator = tl.sum(exponential.to(tl.float32), 1).to(tl.bfloat16)
    probability = (exponential / denominator[:, None]).to(tl.bfloat16)
    result = tl.dot(probability, v, input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, result)


# Ours: kernels bigger or smaller than one 32x32 tile, for pass 4 (shape legalization).
# One 64x64x64 dot: 2x2 output tiles, each a K chain of 2.
@triton.jit
def matmul_64(A, Bm, C):
    r = tl.arange(0, 64)
    off = r[:, None] * 64 + r[None, :]
    tl.store(C + off, tl.dot(tl.load(A + off), tl.load(Bm + off)).to(tl.bfloat16))


# A 64x32 @ 32x32 dot + ReLU: 2 row tiles, no K split.
@triton.jit
def matmul_relu_64x32(A, Bm, C):
    m = tl.arange(0, 64)
    n = tl.arange(0, 32)
    a = tl.load(A + m[:, None] * 32 + n[None, :])
    b = tl.load(Bm + n[:, None] * 32 + n[None, :])
    tl.store(C + m[:, None] * 32 + n[None, :], tl.maximum(tl.dot(a, b), 0.0).to(tl.bfloat16))


# A row sum over 16 columns, scaled: the reduction (loops 64x16) needs padding and is fused into the 64-row scale's
# tiles. (Triton rejects a dot with K < 32, so the same shape of problem comes from a reduction here.)
@triton.jit
def row_sum_scaled_64x16(X, Y):
    m = tl.arange(0, 64)
    n = tl.arange(0, 16)
    x = tl.load(X + m[:, None] * 16 + n[None, :]).to(tl.float32)
    tl.store(Y + m, (tl.sum(x, axis=1) * 2.0).to(tl.bfloat16))


# A softmax over the rows of a 16x16 tile: one load read by a row max, the subtraction and (through exp) a row sum; the division is a reciprocal, which
# is all pass 1 accepts.
# The load is padded to 32x32 once, with one fill, and each reduction masks the lanes that fill does not suit.
@triton.jit
def softmax_16(X, Y):
    r = tl.arange(0, 16)
    x = tl.load(X + r[:, None] * 16 + r[None, :]).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=1)[:, None])
    tl.store(Y + r[:, None] * 16 + r[None, :], (e * (1.0 / tl.sum(e, axis=1))[:, None]).to(tl.bfloat16))


# A softmax over 20 of 32 columns: the load is masked (other = -inf), filled with 0 by the host because the subtraction
# reads it too, so the max reads an add mask and the sum's exp reads one on its input (not a multiply after it).
@triton.jit
def softmax_masked_20(X, Y):
    r = tl.arange(0, 32)
    m = (r < 20)[None, :]
    x = tl.load(X + r[:, None] * 32 + r[None, :], mask=m, other=float("-inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=1)[:, None])
    tl.store(Y + r[:, None] * 32 + r[None, :], (e * (1.0 / tl.sum(e, axis=1))[:, None]).to(tl.bfloat16), mask=m)


# A row max of a 64x64 tile: 2 row tiles, each reducing 2 column tiles.
@triton.jit
def row_max_64(X, Y):
    r = tl.arange(0, 64)
    x = tl.load(X + r[:, None] * 64 + r[None, :]).to(tl.float32)
    tl.store(Y + r, tl.max(x, axis=1).to(tl.bfloat16))


# A 16x16 BF16 add: smaller than a tile, so pass 4 pads it, conversions included.
@triton.jit
def add_16(X, Y):
    r = tl.arange(0, 16)
    off = r[:, None] * 16 + r[None, :]
    x = tl.load(X + off).to(tl.float32)
    tl.store(Y + off, (x + x).to(tl.bfloat16))


# A 50-row BF16 add through a tail mask (rows < 50 of a 64-row block): pass 4 turns the mask into padding.
@triton.jit
def add_rows_50(X, Y):
    r = tl.arange(0, 64)
    c = tl.arange(0, 32)
    off = r[:, None] * 32 + c[None, :]
    mask = (r < 50)[:, None]
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + off, (x + x).to(tl.bfloat16), mask=mask)


# A 64x16 BF16 add: two row tiles, each padded from 16 to 32 columns, stored as a ragged tensor.
@triton.jit
def add_64x16(X, Y):
    r = tl.arange(0, 64)
    c = tl.arange(0, 16)
    off = r[:, None] * 16 + c[None, :]
    x = tl.load(X + off).to(tl.float32)
    tl.store(Y + off, (x + x).to(tl.bfloat16))


# A 16x16 BF16 copy, no compute: the load grows to a tile through a tagged pad, and the store crops it back.
@triton.jit
def copy_16(X, Y):
    r = tl.arange(0, 16)
    off = r[:, None] * 16 + r[None, :]
    tl.store(Y + off, tl.load(X + off))


# A masked BF16 copy of the first 50 rows of a 64x32 block: two row tiles, the second with 18 real rows, no compute.
@triton.jit
def copy_rows_50(X, Y):
    r = tl.arange(0, 64)
    c = tl.arange(0, 32)
    off = r[:, None] * 32 + c[None, :]
    mask = (r < 50)[:, None]
    tl.store(Y + off, tl.load(X + off, mask=mask, other=0.0), mask=mask)


# A K loop whose mask subtracts the loop counter (r < K - k), the usual tail-mask form, for pass 1's subtraction
# rule. Each step reads one contiguous 32x32 tile; with K a multiple of 32 the mask keeps every lane.
@triton.jit
def tail_mask_k_loop(X, Y, K: tl.constexpr):
    r = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for k in range(0, K, 32):
        mask = r[None, :] < K - k
        acc += tl.load(X + k * 32 + r[:, None] * 32 + r[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + r[:, None] * 32 + r[None, :], acc.to(tl.bfloat16))
