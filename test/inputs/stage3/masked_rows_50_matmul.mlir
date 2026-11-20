// Test input: the probe kernel masked_rows_50_matmul after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @masked_rows_50_matmul(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf8E4M3FN>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %cst_1 = arith.constant dense<32> : tensor<64x1xi32>
    %cst_2 = arith.constant dense<50> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = arith.cmpi slt, %0, %cst_2 : tensor<64xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<64x1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %7 = tt.addptr %6, %5 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %8 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %9 = tt.broadcast %7 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x32x!tt.ptr<f8E4M3FN>>
    %10 = tt.broadcast %8 : tensor<1x32xi32> -> tensor<64x32xi32>
    %11 = tt.addptr %9, %10 : tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<64x32xi32>
    %12 = tt.broadcast %3 : tensor<64x1xi1> -> tensor<64x32xi1>
    %13 = tt.load %11, %12, %cst : tensor<64x32x!tt.ptr<f8E4M3FN>>
    %14 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<32x1xi32>
    %16 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %17 = tt.addptr %16, %15 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %18 = tt.broadcast %17 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %19 = tt.broadcast %8 : tensor<1x32xi32> -> tensor<32x32xi32>
    %20 = tt.addptr %18, %19 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %21 = tt.load %20 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %22 = tensor.empty() : tensor<64x32xf32>
    %23 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %24 = linalg.matmul ins(%13, %21 : tensor<64x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%23 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %25 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %26 = tt.addptr %25, %5 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
    %27 = tt.broadcast %26 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x32x!tt.ptr<bf16>>
    %28 = tt.addptr %27, %10 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %29 = tensor.empty() : tensor<64x32xbf16>
    %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<64x32xf32>) outs(%29 : tensor<64x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %31 = arith.truncf %in : f32 to bf16
      linalg.yield %31 : bf16
    } -> tensor<64x32xbf16>
    tt.store %28, %30, %12 : tensor<64x32x!tt.ptr<bf16>>
    tt.return
  }
}
