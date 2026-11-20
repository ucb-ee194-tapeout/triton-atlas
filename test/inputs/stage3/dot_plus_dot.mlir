// Test input: the probe kernel dot_plus_dot after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @dot_plus_dot(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst : tensor<32x1xi32>
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
    %13 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %6 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %15 = tensor.empty() : tensor<32x32xf32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %17 = linalg.matmul ins(%9, %12 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%16 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %18 = linalg.matmul ins(%12, %9 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%17 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %19 = tensor.empty() : tensor<32x32xbf16>
    %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<32x32xf32>) outs(%19 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %21 = arith.truncf %in : f32 to bf16
      linalg.yield %21 : bf16
    } -> tensor<32x32xbf16>
    tt.store %14, %20 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
