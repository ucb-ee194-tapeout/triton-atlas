// Test input: the probe kernel softmax_masked_20 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @softmax_masked_20(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF80> : tensor<32x32xbf16>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32xf32>
    %cst_1 = arith.constant dense<32> : tensor<32x1xi32>
    %cst_2 = arith.constant dense<20> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = arith.cmpi slt, %0, %cst_2 : tensor<32xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<32x1xi32>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %7 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %8 = tt.broadcast %6 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %9 = tt.broadcast %7 : tensor<1x32xi32> -> tensor<32x32xi32>
    %10 = tt.addptr %8, %9 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %11 = tt.broadcast %2 : tensor<1x32xi1> -> tensor<32x32xi1>
    %12 = tt.load %10, %11, %cst : tensor<32x32x!tt.ptr<bf16>>
    %13 = tensor.empty() : tensor<32x32xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<32x32xbf16>) outs(%13 : tensor<32x32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %35 = arith.extf %in : bf16 to f32
      linalg.yield %35 : f32
    } -> tensor<32x32xf32>
    %cst_3 = arith.constant 0xFF800000 : f32
    %15 = tensor.empty() : tensor<32xf32>
    %16 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce { arith.maxnumf } ins(%14 : tensor<32x32xf32>) outs(%16 : tensor<32xf32>) dimensions = [1]
    %17 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%reduced : tensor<32xf32>) outs(%17 : tensor<32x32xf32>) dimensions = [1]
    %18 = tensor.empty() : tensor<32x32xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %35 = arith.subf %in, %in_7 : f32
      linalg.yield %35 : f32
    } -> tensor<32x32xf32>
    %20 = tensor.empty() : tensor<32x32xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<32x32xf32>) outs(%20 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.exp %in : f32
      linalg.yield %35 : f32
    } -> tensor<32x32xf32>
    %22 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %23 = tt.addptr %22, %4 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %24 = tt.broadcast %23 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %25 = tt.addptr %24, %9 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %cst_4 = arith.constant -0.000000e+00 : f32
    %26 = tensor.empty() : tensor<32xf32>
    %27 = linalg.fill ins(%cst_4 : f32) outs(%26 : tensor<32xf32>) -> tensor<32xf32>
    %reduced_5 = linalg.reduce { arith.addf } ins(%21 : tensor<32x32xf32>) outs(%27 : tensor<32xf32>) dimensions = [1]
    %28 = tensor.empty() : tensor<32xf32>
    %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%cst_0, %reduced_5 : tensor<32xf32>, tensor<32xf32>) outs(%28 : tensor<32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %35 = arith.divf %in, %in_7 : f32
      linalg.yield %35 : f32
    } -> tensor<32xf32>
    %30 = tensor.empty() : tensor<32x32xf32>
    %broadcasted_6 = linalg.broadcast ins(%29 : tensor<32xf32>) outs(%30 : tensor<32x32xf32>) dimensions = [1]
    %31 = tensor.empty() : tensor<32x32xf32>
    %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%21, %broadcasted_6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%31 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %35 = arith.mulf %in, %in_7 : f32
      linalg.yield %35 : f32
    } -> tensor<32x32xf32>
    %33 = tensor.empty() : tensor<32x32xbf16>
    %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%32 : tensor<32x32xf32>) outs(%33 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %35 = arith.truncf %in : f32 to bf16
      linalg.yield %35 : bf16
    } -> tensor<32x32xbf16>
    tt.store %25, %34, %11 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
