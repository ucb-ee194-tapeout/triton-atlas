// Fixture: one MLP layer, relu(x @ w.T + b), on one 32x32 tile (stage B's input). TTIR from Triton 3.9.0
// (triton.compile of the kernel below, signature X, W: *fp8e4nv, B, Y: *bf16), debug locations dropped.
//   @triton.jit
//   def mlp(X, W, B, Y):
//       r = tl.arange(0, 32)
//       off = r[:, None] * 32 + r[None, :]
//       x = tl.load(X + off); w = tl.load(W + off); b = tl.load(B + r).to(tl.float32)
//       acc = tl.dot(x, tl.trans(w)) + b[None, :]
//       tl.store(Y + off, tl.maximum(acc, 0.0).to(tl.bfloat16))

module {
  tt.func public @mlp(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<32x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %4 = tt.broadcast %2 : tensor<32x1xi32> -> tensor<32x32xi32>
    %5 = tt.broadcast %3 : tensor<1x32xi32> -> tensor<32x32xi32>
    %6 = arith.addi %4, %5 : tensor<32x32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %8 = tt.addptr %7, %6 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %6 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %12 = tt.load %11 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    %15 = tt.load %14 : tensor<32x!tt.ptr<bf16>>
    %16 = arith.extf %15 : tensor<32xbf16> to tensor<32xf32>
    %17 = tt.trans %12 {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
    %18 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
    %19 = tt.broadcast %18 : tensor<1x32xf32> -> tensor<32x32xf32>
    %20 = tt.dot %9, %17, %19 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    %21 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %22 = tt.addptr %21, %6 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %23 = arith.maxnumf %20, %cst : tensor<32x32xf32>
    %24 = arith.truncf %23 : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %22, %24 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
