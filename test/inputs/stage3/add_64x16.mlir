// Test input: the probe kernel add_64x16 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @add_64x16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = arith.muli %2, %cst : tensor<64x1xi32>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %5 = tt.broadcast %3 : tensor<64x1xi32> -> tensor<64x16xi32>
    %6 = tt.broadcast %4 : tensor<1x16xi32> -> tensor<64x16xi32>
    %7 = arith.addi %5, %6 : tensor<64x16xi32>
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<64x16x!tt.ptr<bf16>>, tensor<64x16xi32>
    %10 = tt.load %9 : tensor<64x16x!tt.ptr<bf16>>
    %11 = tensor.empty() : tensor<64x16xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<64x16xbf16>) outs(%11 : tensor<64x16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %19 = arith.extf %in : bf16 to f32
      linalg.yield %19 : f32
    } -> tensor<64x16xf32>
    %13 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x16x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %7 : tensor<64x16x!tt.ptr<bf16>>, tensor<64x16xi32>
    %15 = tensor.empty() : tensor<64x16xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %12 : tensor<64x16xf32>, tensor<64x16xf32>) outs(%15 : tensor<64x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %19 = arith.addf %in, %in_0 : f32
      linalg.yield %19 : f32
    } -> tensor<64x16xf32>
    %17 = tensor.empty() : tensor<64x16xbf16>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x16xf32>) outs(%17 : tensor<64x16xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %19 = arith.truncf %in : f32 to bf16
      linalg.yield %19 : bf16
    } -> tensor<64x16xbf16>
    tt.store %14, %18 : tensor<64x16x!tt.ptr<bf16>>
    tt.return
  }
}
