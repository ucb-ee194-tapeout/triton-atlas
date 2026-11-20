// Test input: the probe kernel unrolled_by_triton after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @unrolled_by_triton(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<1024> : tensor<32x32xi32>
    %cst_0 = arith.constant dense<32> : tensor<32x32xi32>
    %cst_1 = arith.constant dense<32> : tensor<32x1xi32>
    %cst_2 = arith.constant dense<128> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_2 : tensor<32x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %8 = tt.addptr %6, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %9 = arith.muli %1, %cst_1 : tensor<32x1xi32>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %9 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %12 = tt.broadcast %11 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.addptr %12, %7 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %14 = tt.load %8 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %15 = tt.load %13 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %16 = tensor.empty() : tensor<32x32xf32>
    %17 = linalg.fill ins(%cst_3 : f32) outs(%16 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %18 = linalg.matmul ins(%14, %15 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%17 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %19 = tt.addptr %8, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %20 = tt.addptr %13, %cst : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %21 = tt.load %19 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %22 = tt.load %20 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %23 = linalg.matmul ins(%21, %22 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %24 = tt.addptr %19, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %25 = tt.addptr %20, %cst : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %26 = tt.load %24 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %27 = tt.load %25 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %28 = linalg.matmul ins(%26, %27 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %29 = tt.addptr %24, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %30 = tt.addptr %25, %cst : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %31 = tt.load %29 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %32 = tt.load %30 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %33 = linalg.matmul ins(%31, %32 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%28 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %34 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %35 = tt.addptr %34, %9 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %36 = tt.broadcast %35 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %37 = tt.addptr %36, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %38 = tensor.empty() : tensor<32x32xbf16>
    %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<32x32xf32>) outs(%38 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %40 = arith.truncf %in : f32 to bf16
      linalg.yield %40 : bf16
    } -> tensor<32x32xbf16>
    tt.store %37, %39 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
