// Test input: the probe kernel matmul_relu after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @matmul_relu(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<32x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %8 = tt.addptr %6, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %12 = tt.broadcast %11 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.addptr %12, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %14 = tt.load %13 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %15 = tensor.empty() : tensor<32x32xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %17 = linalg.matmul ins(%9, %14 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%16 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %18 = tensor.empty() : tensor<32x32xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %cst : tensor<32x32xf32>, tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %24 = arith.maxnumf %in, %in_2 : f32
      linalg.yield %24 : f32
    } -> tensor<32x32xf32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %21 = tt.addptr %20, %2 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %22 = tt.broadcast %21 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %23 = tt.addptr %22, %7 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %23, %19 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}
