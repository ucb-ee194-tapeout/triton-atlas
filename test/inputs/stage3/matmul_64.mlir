// Test input: the probe kernel matmul_64 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @matmul_64(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = arith.muli %1, %cst : tensor<64x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = tt.broadcast %2 : tensor<64x1xi32> -> tensor<64x64xi32>
    %5 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<64x64xi32>
    %6 = arith.addi %4, %5 : tensor<64x64xi32>
    %7 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %6 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %10 = tt.addptr %9, %6 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %11 = tt.load %10 : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %12 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %13 = tt.addptr %12, %6 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %14 = tt.load %13 : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %15 = tensor.empty() : tensor<64x64xf32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %17 = linalg.matmul ins(%11, %14 : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>) outs(%16 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %18 = tensor.empty() : tensor<64x64xbf16>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<64x64xf32>) outs(%18 : tensor<64x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %20 = arith.truncf %in : f32 to bf16
      linalg.yield %20 : bf16
    } -> tensor<64x64xbf16>
    tt.store %8, %19 : tensor<64x64x!tt.ptr<bf16>>
    tt.return
  }
}
