// Test input: the probe kernel bias_both_layouts after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it. The bias is added along the rows
// and along the columns, so it is read in both layouts.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @bias_both_layouts(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
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
      %26 = arith.extf %in : bf16 to f32
      linalg.yield %26 : f32
    } -> tensor<32x32xf32>
    %12 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    %14 = tt.load %13 : tensor<32x!tt.ptr<bf16>>
    %15 = tensor.empty() : tensor<32xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%14 : tensor<32xbf16>) outs(%15 : tensor<32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %26 = arith.extf %in : bf16 to f32
      linalg.yield %26 : f32
    } -> tensor<32xf32>
    %17 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%16 : tensor<32xf32>) outs(%17 : tensor<32x32xf32>) dimensions = [1]
    %18 = tensor.empty() : tensor<32x32xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %26 = arith.addf %in, %in_0 : f32
      linalg.yield %26 : f32
    } -> tensor<32x32xf32>
    %b2e = tensor.empty() : tensor<32x32xf32>
    %broadcasted2 = linalg.broadcast ins(%16 : tensor<32xf32>) outs(%b2e : tensor<32x32xf32>) dimensions = [0]
    %b3e = tensor.empty() : tensor<32x32xf32>
    %both = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %broadcasted2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%b3e : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %26 = arith.addf %in, %in_0 : f32
      linalg.yield %26 : f32
    } -> tensor<32x32xf32>
    %20 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %21 = tt.addptr %20, %2 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %22 = tt.broadcast %21 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %23 = tt.addptr %22, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %24 = tensor.empty() : tensor<32x32xbf16>
    %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%both : tensor<32x32xf32>) outs(%24 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %26 = arith.truncf %in : f32 to bf16
      linalg.yield %26 : bf16
    } -> tensor<32x32xbf16>
    tt.store %23, %25 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
