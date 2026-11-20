// RUN: triton-atlas-opt %s --split-input-file --convert-triton-to-atlas-virtual --verify-diagnostics

// Every form pass 7 cannot lower is a named error at the op, never a wrong result.

#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#fa = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fb = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#yf = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  tt.func @mid_chain(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %empty_f8 = tensor.empty() : tensor<32x32xf8E4M3FN>
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
    // expected-error @below {{a K-chain value is used before the chain ends (mid-chain consumer)}}
    %m0 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%ta, %tb : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %m1 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%ta, %tb : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m0 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %bad = linalg.add ins(%m0, %m1 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %bad [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yf]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#fa = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fb = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#yf = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  tt.func @nonzero_accumulator(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %empty_f8 = tensor.empty() : tensor<32x32xf8E4M3FN>
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
    %one = arith.constant 1.000000e+00 : bf16
    %seed = linalg.fill ins(%one : bf16) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // expected-error @below {{a matmul's accumulator must be a zero fill or the previous matmul}}
    %p = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%ta, %tb : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%seed : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %p [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yf]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#fa = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fb = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#yf = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  tt.func @bf16_operands(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %empty_f8 = tensor.empty() : tensor<32x32xf8E4M3FN>
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
    // expected-error @below {{matmul operands must be FP8 tiles}}
    %p = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%acc, %acc : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %p [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yf]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#fa = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fb = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#yf = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @fp8_on_vpu(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %empty_f8 = tensor.empty() : tensor<32x32xf8E4M3FN>
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
    // expected-error @below {{the VPU works on BF16 tiles}}
    %sum = linalg.add ins(%ta, %ta : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%empty_f8 : tensor<32x32xf8E4M3FN>) -> tensor<32x32xf8E4M3FN>
    %e_y = tensor.expand_shape %acc [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yf]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#fa = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#fb = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#yf = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @fp8_store(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %empty_f8 = tensor.empty() : tensor<32x32xf8E4M3FN>
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
    %n = linalg.transpose ins(%ta : tensor<32x32xf8E4M3FN>) outs(%empty_f8 : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
    %e8 = tensor.expand_shape %n [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xf8E4M3FN> into tensor<1x1x32x32xf8E4M3FN>
    %h8 = tensor.empty() : tensor<32x32xf8E4M3FN>
    // expected-error @below {{an FP8 tile cannot be stored}}
    %u8 = linalg.unpack %e8 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h8 {ttatlas.slot = [#yf]} : tensor<1x1x32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
    %e_y = tensor.expand_shape %acc [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yf]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @bf16_transpose(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    // expected-error @below {{a transpose of a BF16 tile has no Atlas lowering}}
    %t = linalg.transpose ins(%x : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) permutation = [1, 0]
    %e_y = tensor.expand_shape %t [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#rowmap = affine_map<(d0, d1) -> (d0)>
module {
  tt.func @broadcast_layout(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %cst_init = arith.constant -0.000000e+00 : bf16
    %init = linalg.fill ins(%cst_init : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %s = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    // expected-error @below {{the broadcast needs the vector in the other layout}}
    %bc = linalg.broadcast ins(%s : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [0]
    %e_y = tensor.expand_shape %bc [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#xv = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = bias_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = cols, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#rowmap = affine_map<(d0, d1) -> (d0)>
module {
  tt.func @layout_mix(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %p_v = tt.splat %b : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %l_v = tt.load %p_v : tensor<32x!tt.ptr<bf16>>
    %g_v = tensor.empty() : tensor<1x32xbf16>
    %k_v = linalg.pack %l_v inner_dims_pos = [0] inner_tiles = [32] into %g_v {ttatlas.slot = [#xv]} : tensor<32xbf16> -> tensor<1x32xbf16>
    %v = tensor.collapse_shape %k_v [[0, 1]] : tensor<1x32xbf16> into tensor<32xbf16>
    %cst_init = arith.constant -0.000000e+00 : bf16
    %init = linalg.fill ins(%cst_init : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %s = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    // expected-error @below {{vector layouts conflict}}
    %t = linalg.add ins(%s, %v : tensor<32xbf16>, tensor<32xbf16>) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %tb = linalg.broadcast ins(%t : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %e_y = tensor.expand_shape %tb [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#rowmap = affine_map<(d0, d1) -> (d0)>
module {
  tt.func @reduction_start(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %cst_init = arith.constant 1.000000e+00 : bf16
    %init = linalg.fill ins(%cst_init : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    // expected-error @below {{a reduction must start from its identity}}
    %s = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %bc = linalg.broadcast ins(%s : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %e_y = tensor.expand_shape %bc [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func @column_reduction(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %cst_init = arith.constant -0.000000e+00 : bf16
    %init = linalg.fill ins(%cst_init : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    // expected-error @below {{only a row reduction}}
    %s = linalg.generic {indexing_maps = [#tile, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%init : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %bc = linalg.broadcast ins(%s : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [0]
    %e_y = tensor.expand_shape %bc [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func @no_vpu_op(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    // expected-error @below {{no Atlas VPU operation}}
    %n = linalg.generic {indexing_maps = [#tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %v = math.sin %in : bf16
      linalg.yield %v : bf16
    } -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %n [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func @not_pointwise(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    // expected-error @below {{pass 7 lowers only pointwise generics}}
    %n = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, #tile], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %v = math.exp %in : bf16
      linalg.yield %v : bf16
    } -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %n [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @cast_between_tiles(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    // expected-error @below {{a cast between compute tiles}}
    %w = arith.extf %x : tensor<32x32xbf16> to tensor<32x32xf32>
    %n = arith.truncf %w : tensor<32x32xf32> to tensor<32x32xbf16>
    %e_y = tensor.expand_shape %n [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @f32_compute(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %empty_wide = tensor.empty() : tensor<32x32xf32>
    %f = arith.constant 0.000000e+00 : f32
    // expected-error @below {{compute must be BF16 or FP8}}
    %w = linalg.fill ins(%f : f32) outs(%empty_wide : tensor<32x32xf32>) -> tensor<32x32xf32>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @store_without_unpack(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    // expected-error @below {{a store needs a linalg.unpack with slots}}
    tt.store %q, %x : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#x2 = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @pack_of_computed(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    %g2 = tensor.empty() : tensor<1x1x32x32xbf16>
    // expected-error @below {{a pack of anything but a load}}
    %k2 = linalg.pack %u_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g2 {ttatlas.slot = [#x2]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#x2 = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @slot_count(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l = tt.load %p : tensor<32x32x!tt.ptr<bf16>>
    %g = tensor.empty() : tensor<1x1x32x32xbf16>
    // expected-error @below {{the pack carries 2 slots for 1 tiles}}
    %k = linalg.pack %l inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g {ttatlas.slot = [#x, #x2]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @duplicate_output(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    %e_z = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_z = tensor.empty() : tensor<32x32xbf16>
    // expected-error @below {{two stores use output index 0}}
    %u_z = linalg.unpack %e_z inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_z {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_z = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_z, %u_z : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @undefined_tile(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %g = tensor.empty() : tensor<1x1x32x32xbf16>
    %h = tensor.empty() : tensor<32x32xbf16>
    %e = tensor.empty() : tensor<32x32xbf16>
    %k = linalg.pack %e inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    // expected-error @below {{a tile of the stored tensor is never computed}}
    %u = linalg.unpack %k inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    tt.return
  }
}

// -----

module {
  tt.func @scalar_pattern_constant(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    // expected-error @below {{a tile constant that is not a splat needs a slot}}
    %v = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]> : tensor<32xbf16>
    %sum = linalg.add ins(%v, %v : tensor<32xbf16>, tensor<32xbf16>) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#x2 = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
    // expected-error @below {{slot indices must be dense from zero}}
  tt.func @input_gap(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x2]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#cst = #ttatlas.slot<index = 1, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32767> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @shared_constant(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    // expected-error @below {{a constant with a slot must have exactly one use}}
    %m = arith.constant {ttatlas.slot = #cst} dense<1.000000e+00> : tensor<32x32xbf16>
    %u1 = linalg.add ins(%x, %m : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %u2 = linalg.add ins(%u1, %m : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %u2 [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#cst = #ttatlas.slot<index = 1, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32767> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = f8E4M3FN>
module {
  tt.func @fp8_constant(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    // expected-error @below {{a constant tile's slot must be a constant slot with a bf16 tile dtype}}
    %m = arith.constant {ttatlas.slot = #cst} dense<0.000000e+00> : tensor<32x32xbf16>
    %u1 = linalg.add ins(%x, %m : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %u1 [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#yt = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @transposed_output(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    // expected-error @below {{an output slot must be a memory slot with a bf16 tile dtype, not transposed}}
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yt]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#yc = #ttatlas.slot<index = 0, kind = constant, role = output_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<0> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @constant_output(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    // expected-error @below {{an output slot must be a memory slot with a bf16 tile dtype, not transposed}}
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#yc]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#rowmap = affine_map<(d0, d1) -> (d0)>
module {
  tt.func @reduction_body(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %cst_init = arith.constant -0.000000e+00 : bf16
    %init = linalg.fill ins(%cst_init : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %k = arith.constant 1.000000e+00 : bf16
    // expected-error @below {{the reduction must combine its accumulator with the input element}}
    %s = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %k : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %bc = linalg.broadcast ins(%s : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %e_y = tensor.expand_shape %bc [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// A slice offset that is an SSA value is not read as 0: slices of tiles must be static.
module {
  tt.func @dynamic_slice_offset(%c: !tt.ptr<bf16>) {
    %one = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : bf16
    %host = tensor.empty() : tensor<64x64xbf16>
    %grid_init = tensor.empty() : tensor<2x2x32x32xbf16>
    %grid = linalg.pack %host inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %grid_init : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
    // expected-error @below {{a slice of tiles must be static with unit strides}}
    %c10 = tensor.extract_slice %grid[%one, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %z = linalg.fill ins(%zero : bf16) outs(%c10 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    tt.return
  }
}
