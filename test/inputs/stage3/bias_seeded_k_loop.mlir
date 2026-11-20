// Test input: the probe kernel bias_seeded_k_loop after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @bias_seeded_k_loop(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>) attributes {noinline = false} {
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
    %14 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %15 = tt.addptr %14, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    %16 = tt.load %15 : tensor<32x!tt.ptr<bf16>>
    %17 = tensor.empty() : tensor<32xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%16 : tensor<32xbf16>) outs(%17 : tensor<32xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %48 = arith.extf %in : bf16 to f32
      linalg.yield %48 : f32
    } -> tensor<32xf32>
    %19 = tensor.empty() : tensor<32x32xf32>
    %broadcasted = linalg.broadcast ins(%18 : tensor<32xf32>) outs(%19 : tensor<32x32xf32>) dimensions = [0]
    %20 = tt.load %8 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %21 = tt.load %13 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %22 = tensor.empty() : tensor<32x32xf32>
    %23 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %24 = linalg.matmul ins(%20, %21 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %25 = tt.addptr %8, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %26 = tt.addptr %13, %cst : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %27 = tt.load %25 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %28 = tt.load %26 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %29 = linalg.matmul ins(%27, %28 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %30 = tt.addptr %25, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %31 = tt.addptr %26, %cst : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %32 = tt.load %30 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %33 = tt.load %31 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %34 = linalg.matmul ins(%32, %33 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%29 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %35 = tt.addptr %30, %cst_0 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %36 = tt.addptr %31, %cst : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %37 = tt.load %35 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %38 = tt.load %36 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %39 = linalg.matmul ins(%37, %38 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%34 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %40 = tensor.empty() : tensor<32x32xf32>
    %41 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%39, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%40 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %48 = arith.addf %in, %in_4 : f32
      linalg.yield %48 : f32
    } -> tensor<32x32xf32>
    %42 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %43 = tt.addptr %42, %9 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %44 = tt.broadcast %43 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %45 = tt.addptr %44, %7 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %46 = tensor.empty() : tensor<32x32xbf16>
    %47 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%41 : tensor<32x32xf32>) outs(%46 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %48 = arith.truncf %in : f32 to bf16
      linalg.yield %48 : bf16
    } -> tensor<32x32xbf16>
    tt.store %45, %47 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
