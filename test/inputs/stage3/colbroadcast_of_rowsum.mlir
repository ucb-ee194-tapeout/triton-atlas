// Test input: the probe kernel colbroadcast_of_rowsum after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @colbroadcast_of_rowsum(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
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
      %23 = arith.extf %in : bf16 to f32
      linalg.yield %23 : f32
    } -> tensor<32x32xf32>
    %cst_0 = arith.constant -0.000000e+00 : f32
    %12 = tensor.empty() : tensor<32xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce { arith.addf } ins(%11 : tensor<32x32xf32>) outs(%13 : tensor<32xf32>) dimensions = [1]
    %14 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%reduced : tensor<32xf32>) outs(%14 : tensor<32x32xf32>) dimensions = [0]
    %15 = tensor.empty() : tensor<32x32xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %23 = arith.addf %in, %in_1 : f32
      linalg.yield %23 : f32
    } -> tensor<32x32xf32>
    %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %18 = tt.addptr %17, %2 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %19 = tt.broadcast %18 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %20 = tt.addptr %19, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %21 = tensor.empty() : tensor<32x32xbf16>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<32x32xf32>) outs(%21 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %23 = arith.truncf %in : f32 to bf16
      linalg.yield %23 : bf16
    } -> tensor<32x32xbf16>
    tt.store %20, %22 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
