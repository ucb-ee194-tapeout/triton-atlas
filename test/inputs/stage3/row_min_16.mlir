// Test input: a padded row min (y[r] = tl.min(x[r, :16])), in the form passes 1-3 give; pass 4 must clamp it with the identity +inf outside the block.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @row_min_16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = arith.muli %1, %cst : tensor<16x1xi32>
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
      %18 = arith.extf %in : bf16 to f32
      linalg.yield %18 : f32
    } -> tensor<16x16xf32>
    %12 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %0 : tensor<16x!tt.ptr<bf16>>, tensor<16xi32>
    %cst_0 = arith.constant 0x7F800000 : f32
    %14 = tensor.empty() : tensor<16xf32>
    %15 = linalg.fill ins(%cst_0 : f32) outs(%14 : tensor<16xf32>) -> tensor<16xf32>
    %reduced = linalg.reduce { arith.minnumf } ins(%11 : tensor<16x16xf32>) outs(%15 : tensor<16xf32>) dimensions = [1]
    %16 = tensor.empty() : tensor<16xbf16>
    %17 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%reduced : tensor<16xf32>) outs(%16 : tensor<16xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %18 = arith.truncf %in : f32 to bf16
      linalg.yield %18 : bf16
    } -> tensor<16xbf16>
    tt.store %13, %17 : tensor<16x!tt.ptr<bf16>>
    tt.return
  }
}
