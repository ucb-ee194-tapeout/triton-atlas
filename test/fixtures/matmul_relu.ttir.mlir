// Fixture: one 32x32 matmul + ReLU (stage A's input). TTIR from the team's ttir2linalg repo, ecd55fd
// matmul_relu_generated.ttir, made by its generate_matmul_relu.py; printed through triton-atlas-opt to drop
// debug locations. Kernel:
//   @triton.jit
//   def matmul_relu_kernel(A, B, C):                       # A, B: *fp8e4nv, C: *fp32
//       rows = tl.arange(0, 32); cols = tl.arange(0, 32)
//       a = tl.load(A + rows[:, None] * 32 + cols[None, :])
//       b = tl.load(B + rows[:, None] * 32 + cols[None, :])
//       result = tl.maximum(tl.dot(a, b, tl.full((32, 32), 0.0, tl.float32)), 0.0)
//       tl.store(C + rows[:, None] * 32 + cols[None, :], result)

module {
  tt.func public @matmul_relu_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<32x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %8 = tt.addptr %6, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %12 = tt.broadcast %11 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.addptr %12, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %14 = tt.load %13 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %15 = tt.dot %9, %14, %cst, inputPrecision = tf32 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    %16 = arith.maxnumf %15, %cst : tensor<32x32xf32>
    %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %18 = tt.addptr %17, %2 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %19 = tt.broadcast %18 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %20 = tt.addptr %19, %7 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %20, %16 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}
