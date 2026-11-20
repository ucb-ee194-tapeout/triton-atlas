// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// Softmax over rows with 20 valid columns, in the form pass 4 leaves it. The masked lanes are -inf in the slot (the host
// writes them), so the max and the sum need no mask. Each exp is clamped at -200 first (the device's exp table is wrong
// far below zero). 1 / sum folds to the reciprocal alone, and the broadcast of a row reduction is no instruction: the
// reduction already puts each row's result across the row.

// CHECK-LABEL: func.func @softmax()
// CHECK: %[[S1:.*]], %[[X:.*]] = "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK: %[[MAX:.*]] = "atlas.virtual_vpu_reduce"(%[[X]]) <{kind = "row_max"}>
// CHECK: %[[D:.*]] = "atlas.virtual_vpu_binary"(%[[X]], %[[MAX]]) <{kind = "sub"}>
// CHECK: %[[FLOOR:.*]] = "atlas.virtual_vpu_splat"() <{bits = 49992 : i32}>
// CHECK: %[[C:.*]] = "atlas.virtual_vpu_binary"(%[[D]], %[[FLOOR]]) <{kind = "max"}>
// CHECK: %[[E:.*]] = "atlas.virtual_vpu_unary"(%[[C]]) <{kind = "exp"}>
// CHECK: %[[SUM:.*]] = "atlas.virtual_vpu_reduce"(%[[E]]) <{kind = "row_sum"}>
// CHECK: %[[R:.*]] = "atlas.virtual_vpu_unary"(%[[SUM]]) <{kind = "recip"}>
// CHECK-NOT: "atlas.virtual_vpu_binary"(%[[SUM]]
// CHECK: %[[Y:.*]] = "atlas.virtual_vpu_binary"(%[[E]], %[[R]]) <{kind = "mul"}>
// CHECK: "atlas.virtual_output_bf16"(%[[S1]], %[[Y]]) <{index = 0 : i32}>

// exp2 is a generic with a math.exp2 body; it takes the same clamp.
// CHECK-LABEL: func.func @softmax_exp2()
// CHECK: "atlas.virtual_vpu_binary"({{.*}}) <{kind = "max"}>
// CHECK-NEXT: "atlas.virtual_vpu_unary"({{.*}}) <{kind = "exp2"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20], other = -inf : bf16, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @softmax(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %cst_init_max = arith.constant 0xFF80 : bf16
    %init_max = linalg.fill ins(%cst_init_max : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %cst_init_sum = arith.constant -0.000000e+00 : bf16
    %init_sum = linalg.fill ins(%cst_init_sum : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %mx = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init_max : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %bm = linalg.broadcast ins(%mx : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %d = linalg.sub ins(%x, %bm : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e = linalg.exp ins(%d : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %sm = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%e : tensor<32x32xbf16>) outs(%init_sum : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %one = arith.constant dense<1.000000e+00> : tensor<32xbf16>
    %rc = linalg.div ins(%one, %sm : tensor<32xbf16>, tensor<32xbf16>) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %rb = linalg.broadcast ins(%rc : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %yv = linalg.mul ins(%e, %rb : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %yv [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
  tt.func @softmax_exp2(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %cst_init_max = arith.constant 0xFF80 : bf16
    %init_max = linalg.fill ins(%cst_init_max : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %cst_init_sum = arith.constant -0.000000e+00 : bf16
    %init_sum = linalg.fill ins(%cst_init_sum : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %mx = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<32x32xbf16>) outs(%init_max : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %bm = linalg.broadcast ins(%mx : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %d = linalg.sub ins(%x, %bm : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e = linalg.generic {indexing_maps = [#tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%d : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %v = math.exp2 %in : bf16
      linalg.yield %v : bf16
    } -> tensor<32x32xbf16>
    %sm = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%e : tensor<32x32xbf16>) outs(%init_sum : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.addf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %one = arith.constant dense<1.000000e+00> : tensor<32xbf16>
    %rc = linalg.div ins(%one, %sm : tensor<32xbf16>, tensor<32xbf16>) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %rb = linalg.broadcast ins(%rc : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %yv = linalg.mul ins(%e, %rb : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e_y = tensor.expand_shape %yv [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
