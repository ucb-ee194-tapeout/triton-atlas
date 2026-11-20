// Test input: the probe kernel rowsum_plus_bias after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @rowsum_plus_bias(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst : tensor<32x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %8 = tt.addptr %6, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<bf16>>
    %10 = tensor.empty() : tensor<32x32xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<32x32xbf16>) outs(%10 : tensor<32x32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %30 = arith.extf %in : bf16 to f32
      linalg.yield %30 : f32
    } -> tensor<32x32xf32>
    %cst_0 = arith.constant -0.000000e+00 : f32
    %12 = tensor.empty() : tensor<32xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce { arith.addf } ins(%11 : tensor<32x32xf32>) outs(%13 : tensor<32xf32>) dimensions = [1]
    %14 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %15 = tt.addptr %14, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    %16 = tt.load %15 : tensor<32x!tt.ptr<bf16>>
    %17 = tensor.empty() : tensor<32xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%16 : tensor<32xbf16>) outs(%17 : tensor<32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %30 = arith.extf %in : bf16 to f32
      linalg.yield %30 : f32
    } -> tensor<32xf32>
    %19 = tensor.empty() : tensor<32xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%reduced, %18 : tensor<32xf32>, tensor<32xf32>) outs(%19 : tensor<32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %30 = arith.addf %in, %in_1 : f32
      linalg.yield %30 : f32
    } -> tensor<32xf32>
    %21 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%20 : tensor<32xf32>) outs(%21 : tensor<32x32xf32>) dimensions = [1]
    %22 = tensor.empty() : tensor<32x32xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%broadcasted, %11 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %30 = arith.addf %in, %in_1 : f32
      linalg.yield %30 : f32
    } -> tensor<32x32xf32>
    %24 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %25 = tt.addptr %24, %2 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %26 = tt.broadcast %25 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %27 = tt.addptr %26, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %28 = tensor.empty() : tensor<32x32xbf16>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<32x32xf32>) outs(%28 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %30 = arith.truncf %in : f32 to bf16
      linalg.yield %30 : bf16
    } -> tensor<32x32xbf16>
    tt.store %27, %29 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
