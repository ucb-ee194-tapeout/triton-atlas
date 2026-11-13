// Test input: a K loop on fixed 32x32 tiles with constant strides, K = 128 (stage C's first input). TTIR from
// Triton 3.9.0 (signature A, Bm: *fp8e4nv, C: *bf16, K = 128), debug locations dropped.
//   @triton.jit
//   def mm32(A, Bm, C, K: tl.constexpr):
//       r = tl.arange(0, 32)
//       a = A + r[:, None] * K + r[None, :]; b = Bm + r[:, None] * 32 + r[None, :]
//       acc = tl.zeros((32, 32), tl.float32)
//       for k in range(0, K // 32):
//           acc = tl.dot(tl.load(a), tl.load(b), acc); a += 32; b += 32 * 32
//       tl.store(C + r[:, None] * 32 + r[None, :], acc.to(tl.bfloat16))

module {
  tt.func public @mm32(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<1024> : tensor<32x32xi32>
    %cst_1 = arith.constant dense<32> : tensor<32x32xi32>
    %cst_2 = arith.constant dense<32> : tensor<32x1xi32>
    %cst_3 = arith.constant dense<128> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_3 : tensor<32x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %8 = tt.addptr %6, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %9 = arith.muli %1, %cst_2 : tensor<32x1xi32>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %9 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %12 = tt.broadcast %11 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.addptr %12, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %14:3 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %8, %arg5 = %13, %arg6 = %cst) -> (tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xf32>)  : i32 {
      %20 = tt.load %arg4 : tensor<32x32x!tt.ptr<f8E4M3FN>>
      %21 = tt.load %arg5 : tensor<32x32x!tt.ptr<f8E4M3FN>>
      %22 = tt.dot %20, %21, %arg6 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
      %23 = tt.addptr %arg4, %cst_1 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
      %24 = tt.addptr %arg5, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
      scf.yield %23, %24, %22 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xf32>
    }
    %15 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %9 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %17 = tt.broadcast %16 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %18 = tt.addptr %17, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %19 = arith.truncf %14#2 : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %18, %19 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
