// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// A one-tile FP8 matmul with a ReLU epilogue, in the form pass 4 leaves it: a pack per load, a matmul on the [n, k]
// weight tile the host writes transposed, a ReLU (max with a zero tile), an unpack to an f32 host tensor. The inputs
// are read where first used, the matmul is one K chain on MXU0, the ReLU is a vector max with a splat.

// CHECK-LABEL: func.func @matmul_relu() -> !atlas.virtual_state
// CHECK-SAME: atlas.input_dram_base = 2415919104 : i64
// CHECK-SAME: atlas.output_dram_base = 2415923200 : i64
// CHECK: %[[S0:.*]] = "atlas.virtual_start"
// CHECK: %[[S1:.*]], %[[A:.*]] = "atlas.virtual_input_fp8"(%[[S0]]) <{index = 0 : i32}>
// CHECK: %[[S2:.*]], %[[B:.*]] = "atlas.virtual_input_fp8"(%[[S1]]) <{index = 1 : i32}>
// CHECK-NOT: atlas.virtual_xlu_transpose
// CHECK: %[[S3:.*]], %[[W:.*]] = "atlas.virtual_mxu_load_weight"(%[[S2]], %[[B]]) <{unit = 0 : i32}>
// CHECK: %[[S4:.*]], %[[ACC:.*]] = "atlas.virtual_mxu_reset"(%[[S3]], %[[A]], %[[W]])
// CHECK: %[[S5:.*]], %[[R:.*]] = "atlas.virtual_mxu_readout_bf16"(%[[S4]], %[[ACC]])
// CHECK: %[[ZERO:.*]] = "atlas.virtual_vpu_splat"() <{bits = 0 : i32}>
// CHECK: %[[RELU:.*]] = "atlas.virtual_vpu_binary"(%[[R]], %[[ZERO]]) <{kind = "max"}>
// CHECK: %[[S6:.*]] = "atlas.virtual_output_bf16"(%[[S5]], %[[RELU]]) <{index = 0 : i32}>
// CHECK: return %[[S6]]
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#a = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#b = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#o = #ttatlas.slot<index = 0, kind = memory, arg = 2, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f32, tile_dtype = bf16>
module {
  tt.func @matmul_relu(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
    %zero_scalar = arith.constant 0.000000e+00 : bf16
    %pa = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %pb = tt.splat %b : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %pc = tt.splat %c : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %xa = tt.load %pa : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %xb = tt.load %pb : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %grid8 = tensor.empty() : tensor<1x1x32x32xf8E4M3FN>
    %packed_a = linalg.pack %xa inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %grid8 {ttatlas.slot = [#a]} : tensor<32x32xf8E4M3FN> -> tensor<1x1x32x32xf8E4M3FN>
    %packed_b = linalg.pack %xb inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %grid8 {ttatlas.slot = [#b]} : tensor<32x32xf8E4M3FN> -> tensor<1x1x32x32xf8E4M3FN>
    %tile_a = tensor.collapse_shape %packed_a [[0, 1, 2], [3]] : tensor<1x1x32x32xf8E4M3FN> into tensor<32x32xf8E4M3FN>
    %tile_b = tensor.collapse_shape %packed_b [[0, 1, 2], [3]] : tensor<1x1x32x32xf8E4M3FN> into tensor<32x32xf8E4M3FN>
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %acc = linalg.fill ins(%zero_scalar : bf16) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %product = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%tile_a, %tile_b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %relu = linalg.generic {indexing_maps = [#tile, #tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%product, %zero : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %other: bf16, %out: bf16):
      %m = arith.maxnumf %in, %other : bf16
      linalg.yield %m : bf16
    } -> tensor<32x32xbf16>
    %grid = tensor.expand_shape %relu [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %host = tensor.empty() : tensor<32x32xbf16>
    %unpacked = linalg.unpack %grid inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %host {ttatlas.slot = [#o]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %wide = arith.extf %unpacked : tensor<32x32xbf16> to tensor<32x32xf32>
    tt.store %pc, %wide : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}
