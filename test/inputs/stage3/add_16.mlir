// Test input: the probe kernel add_16 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @add_16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = arith.muli %1, %cst : tensor<16x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
    %5 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<16x16xi32>
    %6 = arith.addi %4, %5 : tensor<16x16xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %6 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %9 = tt.load %8 : tensor<16x16x!tt.ptr<bf16>>
    %10 = tensor.empty() : tensor<16x16xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<16x16xbf16>) outs(%10 : tensor<16x16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %18 = arith.extf %in : bf16 to f32
      linalg.yield %18 : f32
    } -> tensor<16x16xf32>
    %12 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %6 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %14 = tensor.empty() : tensor<16x16xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %11 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%14 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %18 = arith.addf %in, %in_0 : f32
      linalg.yield %18 : f32
    } -> tensor<16x16xf32>
    %16 = tensor.empty() : tensor<16x16xbf16>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<16x16xf32>) outs(%16 : tensor<16x16xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %18 = arith.truncf %in : f32 to bf16
      linalg.yield %18 : bf16
    } -> tensor<16x16xbf16>
    tt.store %13, %17 : tensor<16x16x!tt.ptr<bf16>>
    tt.return
  }
}
