// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s
// XFAIL: *
// Planned for the next pass 6 PR (reading pass 4 boundary forms). A masked 64x32 load and store (rows < 50), as pass 5 leaves them: pass 6 should give the second row
// tile a slot whose logical shape is the tag's [18, 32]. Today it rejects every masked access with a clear message.
// CHECK: ttatlas.slot<{{.*}}logical_shape = [18, 32]
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @k1(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf8E4M3FN>
    %cst_2 = arith.constant dense<50> : tensor<64x1xi32>
    %cst_3 = arith.constant dense<32> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %4 = arith.muli %2, %cst_3 : tensor<64x1xi32>
    %5 = tt.broadcast %4 : tensor<64x1xi32> -> tensor<64x32xi32>
    %6 = tt.broadcast %3 : tensor<1x32xi32> -> tensor<64x32xi32>
    %7 = arith.addi %5, %6 : tensor<64x32xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x32x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %7 : tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<64x32xi32>
    %10 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %7 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %12 = arith.cmpi slt, %2, %cst_2 : tensor<64x1xi32>
    %13 = tt.broadcast %12 : tensor<64x1xi1> -> tensor<64x32xi1>
    %14 = tt.load %9, %13, %cst_1 {ttatlas.host_dtype = f8E4M3FN} : tensor<64x32x!tt.ptr<f8E4M3FN>>
    %15 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %16 = arith.muli %15, %cst_0 : tensor<32x1xi32>
    %17 = tt.broadcast %16 : tensor<32x1xi32> -> tensor<32x32xi32>
    %18 = tt.broadcast %3 : tensor<1x32xi32> -> tensor<32x32xi32>
    %19 = arith.addi %17, %18 : tensor<32x32xi32>
    %20 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %21 = tt.addptr %20, %19 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %22 = tt.load %21 {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %23 = tensor.empty() : tensor<64x32xbf16>
    %extracted_slice = tensor.extract_slice %14[0, 0] [32, 32] [1, 1] : tensor<64x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %24 = tensor.empty() : tensor<32x32xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %26 = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%extracted_slice, %22 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %27 = tensor.empty() : tensor<32x32xbf16>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<32x32xf32>) outs(%27 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %31 = arith.truncf %in : f32 to bf16
      linalg.yield %31 : bf16
    } -> tensor<32x32xbf16>
    %inserted_slice = tensor.insert_slice %28 into %23[0, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x32xbf16>
    %extracted_slice_4 = tensor.extract_slice %14[32, 0] [32, 32] [1, 1] {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [18, 32]>} : tensor<64x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %29 = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%extracted_slice_4, %22 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%29 : tensor<32x32xf32>) outs(%27 : tensor<32x32xbf16>) attrs =  {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [18, 32]>} {
    ^bb0(%in: f32, %out: bf16):
      %31 = arith.truncf %in : f32 to bf16
      linalg.yield %31 : bf16
    } -> tensor<32x32xbf16>
    %inserted_slice_5 = tensor.insert_slice %30 into %inserted_slice[32, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x32xbf16>
    tt.store %11, %inserted_slice_5, %13 {ttatlas.host_dtype = bf16} : tensor<64x32x!tt.ptr<bf16>>
    tt.return
  }
}
