// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s
// XFAIL: *
// Planned for the next pass 6 PR (reading pass 4 boundary forms). A 16x16 load read through a tagged tensor.pad and a 16x16 stored crop, as pass 5 leaves them: pass 6
// should give the load and the store slots with logical shape [16, 16]. Today it rejects the sub-tile access.
// CHECK: ttatlas.slot<{{.*}}logical_shape = [16, 16]
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @k6(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %3 = arith.muli %1, %cst_0 : tensor<16x1xi32>
    %4 = tt.broadcast %3 : tensor<16x1xi32> -> tensor<16x16xi32>
    %5 = tt.broadcast %2 : tensor<1x16xi32> -> tensor<16x16xi32>
    %6 = arith.addi %4, %5 : tensor<16x16xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %6 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %9 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %10 = tt.addptr %9, %6 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %11 = tt.load %8 {ttatlas.host_dtype = bf16} : tensor<16x16x!tt.ptr<bf16>>
    %padded = tensor.pad %11 low[0, 0] high[16, 16] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : bf16
    } {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 16]>} : tensor<16x16xbf16> to tensor<32x32xbf16>
    %12 = tensor.empty() : tensor<32x32xbf16>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%padded, %padded : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%12 : tensor<32x32xbf16>) attrs =  {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 16]>} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %14 = arith.addf %in, %in_1 : bf16
      linalg.yield %14 : bf16
    } -> tensor<32x32xbf16>
    %extracted_slice = tensor.extract_slice %13[0, 0] [16, 16] [1, 1] : tensor<32x32xbf16> to tensor<16x16xbf16>
    tt.store %10, %extracted_slice {ttatlas.host_dtype = bf16} : tensor<16x16x!tt.ptr<bf16>>
    tt.return
  }
}
