// Test input: the probe kernel matmul_relu_64x32 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @matmul_relu_64x32(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %cst_1 = arith.constant dense<32> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = arith.muli %2, %cst_1 : tensor<64x1xi32>
    %4 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x32x!tt.ptr<f8E4M3FN>>
    %8 = tt.broadcast %6 : tensor<1x32xi32> -> tensor<64x32xi32>
    %9 = tt.addptr %7, %8 : tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<64x32xi32>
    %10 = tt.load %9 : tensor<64x32x!tt.ptr<f8E4M3FN>>
    %11 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %12 = arith.muli %11, %cst_0 : tensor<32x1xi32>
    %13 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %14 = tt.addptr %13, %12 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %15 = tt.broadcast %14 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %16 = tt.broadcast %6 : tensor<1x32xi32> -> tensor<32x32xi32>
    %17 = tt.addptr %15, %16 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %18 = tt.load %17 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %19 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %20 = tt.addptr %19, %3 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
    %21 = tt.broadcast %20 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x32x!tt.ptr<bf16>>
    %22 = tt.addptr %21, %8 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %23 = tensor.empty() : tensor<64x32xf32>
    %24 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %25 = linalg.matmul ins(%10, %18 : tensor<64x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%24 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %26 = tensor.empty() : tensor<64x32xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%25, %cst : tensor<64x32xf32>, tensor<64x32xf32>) outs(%26 : tensor<64x32xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %30 = arith.maxnumf %in, %in_3 : f32
      linalg.yield %30 : f32
    } -> tensor<64x32xf32>
    %28 = tensor.empty() : tensor<64x32xbf16>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<64x32xf32>) outs(%28 : tensor<64x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %30 = arith.truncf %in : f32 to bf16
      linalg.yield %30 : bf16
    } -> tensor<64x32xbf16>
    tt.store %22, %29 : tensor<64x32x!tt.ptr<bf16>>
    tt.return
  }
}
