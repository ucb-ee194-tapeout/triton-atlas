// Test input: the probe kernel tail_mask_k_loop after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @tail_mask_k_loop(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<32> : tensor<1x32xi32>
    %cst_0 = arith.constant dense<64> : tensor<1x32xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
    %cst_3 = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %2 = arith.cmpi slt, %1, %cst_0 : tensor<1x32xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %4 = arith.muli %3, %cst_3 : tensor<32x1xi32>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %7 = tt.broadcast %6 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %8 = tt.broadcast %1 : tensor<1x32xi32> -> tensor<32x32xi32>
    %9 = tt.addptr %7, %8 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %10 = tt.broadcast %2 : tensor<1x32xi1> -> tensor<32x32xi1>
    %11 = tt.load %9, %10, %cst_2 : tensor<32x32x!tt.ptr<bf16>>
    %12 = tensor.empty() : tensor<32x32xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<32x32xbf16>) outs(%12 : tensor<32x32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %34 = arith.extf %in : bf16 to f32
      linalg.yield %34 : f32
    } -> tensor<32x32xf32>
    %14 = tensor.empty() : tensor<32x32xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %cst_1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%14 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %34 = arith.addf %in, %in_4 : f32
      linalg.yield %34 : f32
    } -> tensor<32x32xf32>
    %16 = arith.cmpi slt, %1, %cst : tensor<1x32xi32>
    %17 = tt.addptr %arg0, %c1024_i32 : !tt.ptr<bf16>, i32
    %18 = tt.splat %17 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %19 = tt.addptr %18, %4 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %20 = tt.broadcast %19 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %21 = tt.addptr %20, %8 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %22 = tt.broadcast %16 : tensor<1x32xi1> -> tensor<32x32xi1>
    %23 = tt.load %21, %22, %cst_2 : tensor<32x32x!tt.ptr<bf16>>
    %24 = tensor.empty() : tensor<32x32xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<32x32xbf16>) outs(%24 : tensor<32x32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %34 = arith.extf %in : bf16 to f32
      linalg.yield %34 : f32
    } -> tensor<32x32xf32>
    %26 = tensor.empty() : tensor<32x32xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%15, %25 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %34 = arith.addf %in, %in_4 : f32
      linalg.yield %34 : f32
    } -> tensor<32x32xf32>
    %28 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %29 = tt.addptr %28, %4 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %30 = tt.broadcast %29 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %32 = tensor.empty() : tensor<32x32xbf16>
    %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<32x32xf32>) outs(%32 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %34 = arith.truncf %in : f32 to bf16
      linalg.yield %34 : bf16
    } -> tensor<32x32xbf16>
    tt.store %31, %33 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
