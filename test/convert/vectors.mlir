// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// A 32-vector is one tile too. A row reduction gives one value per row (layout rows), and a vector that is added to it
// or broadcast must have the same layout. A bias that the host repeats across the row (rows) adds to the reduction
// directly, and a broadcast then spreads it over the columns without any instruction. A column bias (cols) is added
// to a tile after its broadcast along the rows.

// CHECK-LABEL: func.func @rowsum_plus_bias()
// CHECK: %[[S1:.*]], %[[X:.*]] = "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: %[[SUM:.*]] = "atlas.virtual_vpu_reduce"(%[[X]]) <{kind = "row_sum"}>
// CHECK-NEXT: %[[S2:.*]], %[[B:.*]] = "atlas.virtual_input_bf16"(%[[S1]]) <{index = 1 : i32}>
// CHECK-NEXT: %[[T:.*]] = "atlas.virtual_vpu_binary"(%[[SUM]], %[[B]]) <{kind = "add"}>
// CHECK-NEXT: %[[Y:.*]] = "atlas.virtual_vpu_binary"(%[[T]], %[[X]]) <{kind = "add"}>
// CHECK-NEXT: "atlas.virtual_output_bf16"(%[[S2]], %[[Y]]) <{index = 0 : i32}>

// CHECK-LABEL: func.func @column_bias()
// CHECK: %[[S1:.*]], %[[X:.*]] = "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: %[[S2:.*]], %[[B:.*]] = "atlas.virtual_input_bf16"(%[[S1]]) <{index = 1 : i32}>
// CHECK-NEXT: "atlas.virtual_vpu_binary"(%[[X]], %[[B]]) <{kind = "add"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#bias = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = bias_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = rows, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#biasc = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = bias_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = cols, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 2, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @rowsum_plus_bias(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %c: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %cst_init_sum = arith.constant -0.000000e+00 : bf16
    %init_sum = linalg.fill ins(%cst_init_sum : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %p_v = tt.splat %b : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %l_v = tt.load %p_v : tensor<32x!tt.ptr<bf16>>
    %g_v = tensor.empty() : tensor<1x32xbf16>
    %k_v = linalg.pack %l_v inner_dims_pos = [0] inner_tiles = [32] into %g_v {ttatlas.slot = [#bias]} : tensor<32xbf16> -> tensor<1x32xbf16>
    %v = tensor.collapse_shape %k_v [[0, 1]] : tensor<1x32xbf16> into tensor<32xbf16>
    %s = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init_sum : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %t = linalg.add ins(%s, %v : tensor<32xbf16>, tensor<32xbf16>) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %tb = linalg.broadcast ins(%t : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %yv = linalg.add ins(%tb, %x : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %yv [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
  tt.func @column_bias(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %c: !tt.ptr<bf16>) {
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
    %k_v = linalg.pack %l_v inner_dims_pos = [0] inner_tiles = [32] into %g_v {ttatlas.slot = [#biasc]} : tensor<32xbf16> -> tensor<1x32xbf16>
    %v = tensor.collapse_shape %k_v [[0, 1]] : tensor<1x32xbf16> into tensor<32xbf16>
    %tb = linalg.broadcast ins(%v : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [0]
    %yv = linalg.add ins(%x, %tb : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %yv [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %c : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
