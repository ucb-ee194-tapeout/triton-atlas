// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// The MXU takes its weight as [n, k]. A weight whose matmul reads it as [k, n] is transposed on the XLU first, and so is
// a weight that a linalg.transpose of an FP8 tile feeds; the host writes a weight tile transposed when it can, and then
// nothing is emitted (see matmul-relu).

// CHECK-LABEL: func.func @weight_kn()
// CHECK: %[[S:.*]], %[[W:.*]] = "atlas.virtual_input_fp8"(%{{.*}}) <{index = 1 : i32}>
// CHECK: %[[T:.*]] = "atlas.virtual_xlu_transpose"(%[[W]])
// CHECK: "atlas.virtual_mxu_load_weight"(%{{.*}}, %[[T]])

// CHECK-LABEL: func.func @explicit_transpose()
// CHECK: %[[S:.*]], %[[W:.*]] = "atlas.virtual_input_fp8"(%{{.*}}) <{index = 1 : i32}>
// CHECK: %[[T:.*]] = "atlas.virtual_xlu_transpose"(%[[W]])
// CHECK: "atlas.virtual_mxu_load_weight"(%{{.*}}, %[[T]])
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#fa = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fb = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_kn, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fo = #ttatlas.slot<index = 0, kind = memory, arg = 2, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @weight_kn(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_ta = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %l_ta = tt.load %p_ta : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %g_ta = tensor.empty() : tensor<1x1x32x32xf8E4M3FN>
    %k_ta = linalg.pack %l_ta inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_ta {ttatlas.slot = [#fa]} : tensor<32x32xf8E4M3FN> -> tensor<1x1x32x32xf8E4M3FN>
    %ta = tensor.collapse_shape %k_ta [[0, 1, 2], [3]] : tensor<1x1x32x32xf8E4M3FN> into tensor<32x32xf8E4M3FN>
    %p_tb = tt.splat %b : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %l_tb = tt.load %p_tb : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %g_tb = tensor.empty() : tensor<1x1x32x32xf8E4M3FN>
    %k_tb = linalg.pack %l_tb inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_tb {ttatlas.slot = [#fb]} : tensor<32x32xf8E4M3FN> -> tensor<1x1x32x32xf8E4M3FN>
    %tb = tensor.collapse_shape %k_tb [[0, 1, 2], [3]] : tensor<1x1x32x32xf8E4M3FN> into tensor<32x32xf8E4M3FN>
    %cst_acc = arith.constant 0.000000e+00 : bf16
    %acc = linalg.fill ins(%cst_acc : bf16) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %p = linalg.matmul indexing_maps = [#map, #map1kn, #map2] ins(%ta, %tb : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %p [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#fo]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
  tt.func @explicit_transpose(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_f8 = tensor.empty() : tensor<32x32xf8E4M3FN>
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_ta = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %l_ta = tt.load %p_ta : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %g_ta = tensor.empty() : tensor<1x1x32x32xf8E4M3FN>
    %k_ta = linalg.pack %l_ta inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_ta {ttatlas.slot = [#fa]} : tensor<32x32xf8E4M3FN> -> tensor<1x1x32x32xf8E4M3FN>
    %ta = tensor.collapse_shape %k_ta [[0, 1, 2], [3]] : tensor<1x1x32x32xf8E4M3FN> into tensor<32x32xf8E4M3FN>
    %p_tb = tt.splat %b : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %l_tb = tt.load %p_tb : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %g_tb = tensor.empty() : tensor<1x1x32x32xf8E4M3FN>
    %k_tb = linalg.pack %l_tb inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_tb {ttatlas.slot = [#fb]} : tensor<32x32xf8E4M3FN> -> tensor<1x1x32x32xf8E4M3FN>
    %tb = tensor.collapse_shape %k_tb [[0, 1, 2], [3]] : tensor<1x1x32x32xf8E4M3FN> into tensor<32x32xf8E4M3FN>
    %tw = linalg.transpose ins(%tb : tensor<32x32xf8E4M3FN>) outs(%empty_f8 : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
    %cst_acc = arith.constant 0.000000e+00 : bf16
    %acc = linalg.fill ins(%cst_acc : bf16) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %p = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%ta, %tw : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %p [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#fo]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
