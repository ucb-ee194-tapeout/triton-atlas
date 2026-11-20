// Test input: the probe kernel softmax_16 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @softmax_16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %2 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %6 = tt.broadcast %4 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x16x!tt.ptr<bf16>>
    %7 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<16x16xi32>
    %8 = tt.addptr %6, %7 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %9 = tt.load %8 : tensor<16x16x!tt.ptr<bf16>>
    %10 = tensor.empty() : tensor<16x16xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<16x16xbf16>) outs(%10 : tensor<16x16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %32 = arith.extf %in : bf16 to f32
      linalg.yield %32 : f32
    } -> tensor<16x16xf32>
    %cst_1 = arith.constant 0xFF800000 : f32
    %12 = tensor.empty() : tensor<16xf32>
    %13 = linalg.fill ins(%cst_1 : f32) outs(%12 : tensor<16xf32>) -> tensor<16xf32>
    %reduced = linalg.reduce { arith.maxnumf } ins(%11 : tensor<16x16xf32>) outs(%13 : tensor<16xf32>) dimensions = [1]
    %14 = tensor.empty() : tensor<16x16xf32>
    %broadcasted = linalg.broadcast ins(%reduced : tensor<16xf32>) outs(%14 : tensor<16x16xf32>) dimensions = [1]
    %15 = tensor.empty() : tensor<16x16xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %broadcasted : tensor<16x16xf32>, tensor<16x16xf32>) outs(%15 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %32 = arith.subf %in, %in_5 : f32
      linalg.yield %32 : f32
    } -> tensor<16x16xf32>
    %17 = tensor.empty() : tensor<16x16xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<16x16xf32>) outs(%17 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = math.exp %in : f32
      linalg.yield %32 : f32
    } -> tensor<16x16xf32>
    %19 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %20 = tt.addptr %19, %2 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi32>
    %21 = tt.broadcast %20 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x16x!tt.ptr<bf16>>
    %22 = tt.addptr %21, %7 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %cst_2 = arith.constant -0.000000e+00 : f32
    %23 = tensor.empty() : tensor<16xf32>
    %24 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<16xf32>) -> tensor<16xf32>
    %reduced_3 = linalg.reduce { arith.addf } ins(%18 : tensor<16x16xf32>) outs(%24 : tensor<16xf32>) dimensions = [1]
    %25 = tensor.empty() : tensor<16xf32>
    %26 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%cst, %reduced_3 : tensor<16xf32>, tensor<16xf32>) outs(%25 : tensor<16xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %32 = arith.divf %in, %in_5 : f32
      linalg.yield %32 : f32
    } -> tensor<16xf32>
    %27 = tensor.empty() : tensor<16x16xf32>
    %broadcasted_4 = linalg.broadcast ins(%26 : tensor<16xf32>) outs(%27 : tensor<16x16xf32>) dimensions = [1]
    %28 = tensor.empty() : tensor<16x16xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %broadcasted_4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%28 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %32 = arith.mulf %in, %in_5 : f32
      linalg.yield %32 : f32
    } -> tensor<16x16xf32>
    %30 = tensor.empty() : tensor<16x16xbf16>
    %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%29 : tensor<16x16xf32>) outs(%30 : tensor<16x16xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %32 = arith.truncf %in : f32 to bf16
      linalg.yield %32 : bf16
    } -> tensor<16x16xbf16>
    tt.store %22, %31 : tensor<16x16x!tt.ptr<bf16>>
    tt.return
  }
}
