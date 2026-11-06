// RUN: triton-atlas-opt %s --ttatlas-assign-slots --ttatlas-verify-contract=stage=5 | FileCheck %s
// CHECK: index = 1, arg = 1, role = weight_fp8_nk
module {
  tt.func public @matmul_relu_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %9 = tt.load %8 {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %12 = tt.broadcast %11 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.addptr %12, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %14 = tt.load %13 {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %bt = tt.trans %14 {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
    %15 = tt.dot %9, %bt, %cst, inputPrecision = tf32 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    %16 = arith.maxnumf %15, %cst : tensor<32x32xf32>
    %17 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %18 = tt.addptr %17, %2 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %19 = tt.broadcast %18 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %20 = tt.addptr %19, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %narrow = arith.truncf %16 : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %20, %narrow {ttatlas.host_dtype = f32} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
