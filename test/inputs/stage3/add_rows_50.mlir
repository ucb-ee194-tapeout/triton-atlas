// Test input: the probe kernel add_rows_50 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @add_rows_50(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xbf16>
    %cst_0 = arith.constant dense<50> : tensor<64xi32>
    %cst_1 = arith.constant dense<32> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = arith.muli %2, %cst_1 : tensor<64x1xi32>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = tt.broadcast %3 : tensor<64x1xi32> -> tensor<64x32xi32>
    %6 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<64x32xi32>
    %7 = arith.addi %5, %6 : tensor<64x32xi32>
    %8 = arith.cmpi slt, %0, %cst_0 : tensor<64xi32>
    %9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %10 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %7 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %12 = tt.broadcast %9 : tensor<64x1xi1> -> tensor<64x32xi1>
    %13 = tt.load %11, %12, %cst : tensor<64x32x!tt.ptr<bf16>>
    %14 = tensor.empty() : tensor<64x32xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<64x32xbf16>) outs(%14 : tensor<64x32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %22 = arith.extf %in : bf16 to f32
      linalg.yield %22 : f32
    } -> tensor<64x32xf32>
    %16 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %17 = tt.addptr %16, %7 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %18 = tensor.empty() : tensor<64x32xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%15, %15 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%18 : tensor<64x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %22 = arith.addf %in, %in_2 : f32
      linalg.yield %22 : f32
    } -> tensor<64x32xf32>
    %20 = tensor.empty() : tensor<64x32xbf16>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<64x32xf32>) outs(%20 : tensor<64x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %22 = arith.truncf %in : f32 to bf16
      linalg.yield %22 : bf16
    } -> tensor<64x32xbf16>
    tt.store %17, %21, %12 : tensor<64x32x!tt.ptr<bf16>>
    tt.return
  }
}
