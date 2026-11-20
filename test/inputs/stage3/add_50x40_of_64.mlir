// Test input: the probe kernel add_50x40_of_64 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @add_50x40_of_64(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xbf16>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
    %cst_1 = arith.constant dense<40> : tensor<64xi32>
    %cst_2 = arith.constant dense<50> : tensor<64xi32>
    %cst_3 = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = arith.muli %1, %cst_3 : tensor<64x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = tt.broadcast %2 : tensor<64x1xi32> -> tensor<64x64xi32>
    %5 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<64x64xi32>
    %6 = arith.addi %4, %5 : tensor<64x64xi32>
    %7 = arith.cmpi slt, %0, %cst_2 : tensor<64xi32>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %9 = arith.cmpi slt, %0, %cst_1 : tensor<64xi32>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %11 = tt.broadcast %8 : tensor<64x1xi1> -> tensor<64x64xi1>
    %12 = tt.broadcast %10 : tensor<1x64xi1> -> tensor<64x64xi1>
    %13 = arith.andi %11, %12 : tensor<64x64xi1>
    %14 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %15 = tt.addptr %14, %6 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %16 = tt.load %15, %13, %cst : tensor<64x64x!tt.ptr<bf16>>
    %17 = tensor.empty() : tensor<64x64xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x64xbf16>) outs(%17 : tensor<64x64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %25 = arith.extf %in : bf16 to f32
      linalg.yield %25 : f32
    } -> tensor<64x64xf32>
    %19 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %20 = tt.addptr %19, %6 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %21 = tensor.empty() : tensor<64x64xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %cst_0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%21 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %25 = arith.addf %in, %in_4 : f32
      linalg.yield %25 : f32
    } -> tensor<64x64xf32>
    %23 = tensor.empty() : tensor<64x64xbf16>
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%22 : tensor<64x64xf32>) outs(%23 : tensor<64x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %25 = arith.truncf %in : f32 to bf16
      linalg.yield %25 : bf16
    } -> tensor<64x64xbf16>
    tt.store %20, %24, %13 : tensor<64x64x!tt.ptr<bf16>>
    tt.return
  }
}
