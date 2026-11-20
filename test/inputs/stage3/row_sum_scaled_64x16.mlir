// Test input: the probe kernel row_sum_scaled_64x16 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @row_sum_scaled_64x16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<2.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<16> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = arith.muli %2, %cst_0 : tensor<64x1xi32>
    %4 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x16x!tt.ptr<bf16>>
    %8 = tt.broadcast %6 : tensor<1x16xi32> -> tensor<64x16xi32>
    %9 = tt.addptr %7, %8 : tensor<64x16x!tt.ptr<bf16>>, tensor<64x16xi32>
    %10 = tt.load %9 : tensor<64x16x!tt.ptr<bf16>>
    %11 = tensor.empty() : tensor<64x16xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<64x16xbf16>) outs(%11 : tensor<64x16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %21 = arith.extf %in : bf16 to f32
      linalg.yield %21 : f32
    } -> tensor<64x16xf32>
    %13 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %0 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
    %cst_1 = arith.constant -0.000000e+00 : f32
    %15 = tensor.empty() : tensor<64xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce { arith.addf } ins(%12 : tensor<64x16xf32>) outs(%16 : tensor<64xf32>) dimensions = [1]
    %17 = tensor.empty() : tensor<64xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%reduced, %cst : tensor<64xf32>, tensor<64xf32>) outs(%17 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %21 = arith.mulf %in, %in_2 : f32
      linalg.yield %21 : f32
    } -> tensor<64xf32>
    %19 = tensor.empty() : tensor<64xbf16>
    %20 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%18 : tensor<64xf32>) outs(%19 : tensor<64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %21 = arith.truncf %in : f32 to bf16
      linalg.yield %21 : bf16
    } -> tensor<64xbf16>
    tt.store %14, %20 : tensor<64x!tt.ptr<bf16>>
    tt.return
  }
}
