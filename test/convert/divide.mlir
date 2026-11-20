// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// x / y is x * recip(y), and 1 / y is the reciprocal alone. Both forms of the divide reach pass 7: the named linalg.div
// and the generic with an arith.divf body.

// CHECK-LABEL: func.func @divide()
// CHECK: %[[S1:.*]], %[[X:.*]] = "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: %[[S2:.*]], %[[Y:.*]] = "atlas.virtual_input_bf16"(%[[S1]]) <{index = 1 : i32}>
// CHECK-NEXT: %[[R:.*]] = "atlas.virtual_vpu_unary"(%[[Y]]) <{kind = "recip"}>
// CHECK-NEXT: %[[Q:.*]] = "atlas.virtual_vpu_binary"(%[[X]], %[[R]]) <{kind = "mul"}>
// CHECK-NEXT: "atlas.virtual_output_bf16"(%[[S2]], %[[Q]]) <{index = 0 : i32}>

// CHECK-LABEL: func.func @one_over_y()
// CHECK: "atlas.virtual_input_bf16"
// CHECK-NEXT: %[[R:.*]] = "atlas.virtual_vpu_unary"({{.*}}) <{kind = "recip"}>
// CHECK-NEXT: "atlas.virtual_output_bf16"({{.*}}, %[[R]])
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y0 = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#o = #ttatlas.slot<index = 0, kind = memory, arg = 2, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @divide(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %p_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_y = tt.load %p_y : tensor<32x32x!tt.ptr<bf16>>
    %g_y = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_y = linalg.pack %l_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_y {ttatlas.slot = [#y]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %y = tensor.collapse_shape %k_y [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %q = linalg.div ins(%x, %y : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %q [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#o]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
  tt.func @one_over_y(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %p_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_y = tt.load %p_y : tensor<32x32x!tt.ptr<bf16>>
    %g_y = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_y = linalg.pack %l_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_y {ttatlas.slot = [#y0]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %y = tensor.collapse_shape %k_y [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %one = arith.constant 1.000000e+00 : bf16
    %q = linalg.generic {indexing_maps = [#tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%y : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %d = arith.divf %one, %in : bf16
      linalg.yield %d : bf16
    } -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %q [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#o]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
