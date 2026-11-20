// Test input: the probe kernel row_max_64 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  tt.func public @row_max_64(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = arith.muli %1, %cst : tensor<64x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %2 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.broadcast %4 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x64x!tt.ptr<bf16>>
    %7 = tt.broadcast %5 : tensor<1x64xi32> -> tensor<64x64xi32>
    %8 = tt.addptr %6, %7 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %9 = tt.load %8 : tensor<64x64x!tt.ptr<bf16>>
    %10 = tensor.empty() : tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<64x64xbf16>) outs(%10 : tensor<64x64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %18 = arith.extf %in : bf16 to f32
      linalg.yield %18 : f32
    } -> tensor<64x64xf32>
    %12 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %0 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
    %cst_0 = arith.constant 0xFF800000 : f32
    %14 = tensor.empty() : tensor<64xf32>
    %15 = linalg.fill ins(%cst_0 : f32) outs(%14 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce { arith.maxnumf } ins(%11 : tensor<64x64xf32>) outs(%15 : tensor<64xf32>) dimensions = [1]
    %16 = tensor.empty() : tensor<64xbf16>
    %17 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%reduced : tensor<64xf32>) outs(%16 : tensor<64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %18 = arith.truncf %in : f32 to bf16
      linalg.yield %18 : bf16
    } -> tensor<64xbf16>
    tt.store %13, %17 : tensor<64x!tt.ptr<bf16>>
    tt.return
  }
}
