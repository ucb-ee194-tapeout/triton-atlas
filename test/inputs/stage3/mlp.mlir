// Test input: the probe kernel mlp after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
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
    %16 = tensor.empty() : tensor<32xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%15 : tensor<32xbf16>) outs(%16 : tensor<32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %31 = arith.extf %in : bf16 to f32
      linalg.yield %31 : f32
    } -> tensor<32xf32>
    %18 = tensor.empty() : tensor<32x32xf8E4M3FN>
    %transposed = linalg.transpose ins(%12 : tensor<32x32xf8E4M3FN>) outs(%18 : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
    %19 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%17 : tensor<32xf32>) outs(%19 : tensor<32x32xf32>) dimensions = [0]
    %cst_1 = arith.constant 0.000000e+00 : f32
    %20 = tensor.empty() : tensor<32x32xf32>
    %21 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %22 = linalg.matmul ins(%9, %transposed : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %23 = tensor.empty() : tensor<32x32xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %31 = arith.addf %in, %in_2 : f32
      linalg.yield %31 : f32
    } -> tensor<32x32xf32>
    %25 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %26 = tt.addptr %25, %6 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %27 = tensor.empty() : tensor<32x32xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%24, %cst : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %31 = arith.maxnumf %in, %in_2 : f32
      linalg.yield %31 : f32
    } -> tensor<32x32xf32>
    %29 = tensor.empty() : tensor<32x32xbf16>
    %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%28 : tensor<32x32xf32>) outs(%29 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %31 = arith.truncf %in : f32 to bf16
      linalg.yield %31 : bf16
    } -> tensor<32x32xbf16>
    tt.store %26, %30 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
