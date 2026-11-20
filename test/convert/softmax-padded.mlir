// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// Softmax of a 16x16 block: the padded lanes would reach the max and the sum, so pass 4 puts a clamp (a minnumf and a
// maxnumf with constant tiles) in front of each reader. A clamp constant is an input slot the host writes (0x7fff inside
// the block, the reader's identity outside); nothing is computed to make it. The inputs are read where the first reader
// needs them: the data tile, the max's clamp, then the sum's two.

// CHECK-LABEL: func.func @softmax_padded()
// CHECK-SAME: \22kind\22:\22constant\22
// CHECK: "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: "atlas.virtual_input_bf16"(%{{.*}}) <{index = 1 : i32}>
// CHECK-NEXT: "atlas.virtual_vpu_binary"({{.*}}) <{kind = "min"}>
// CHECK-NEXT: "atlas.virtual_vpu_reduce"({{.*}}) <{kind = "row_max"}>
// CHECK: "atlas.virtual_vpu_unary"({{.*}}) <{kind = "exp"}>
// CHECK: "atlas.virtual_input_bf16"(%{{.*}}) <{index = 2 : i32}>
// CHECK-NEXT: "atlas.virtual_vpu_binary"({{.*}}) <{kind = "min"}>
// CHECK-NEXT: "atlas.virtual_input_bf16"(%{{.*}}) <{index = 3 : i32}>
// CHECK-NEXT: "atlas.virtual_vpu_binary"({{.*}}) <{kind = "max"}>
// CHECK-NEXT: "atlas.virtual_vpu_reduce"({{.*}}) <{kind = "row_sum"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#x = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 16, col_stride = 1, block = [0, 0, 16, 16], valid = [0, 0, 16, 16], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 16, col_stride = 1, block = [0, 0, 16, 16], valid = [0, 0, 16, 16], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#hi1 = #ttatlas.slot<index = 1, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32767> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#hi2 = #ttatlas.slot<index = 2, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32767> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#lo2 = #ttatlas.slot<index = 3, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<65535> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @softmax_padded(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %hi1 = arith.constant {ttatlas.slot = #hi1} dense<0.000000e+00> : tensor<32x32xbf16>
    %hi2 = arith.constant {ttatlas.slot = #hi2} dense<0.000000e+00> : tensor<32x32xbf16>
    %lo2 = arith.constant {ttatlas.slot = #lo2} dense<0.000000e+00> : tensor<32x32xbf16>
    %cst_init_max = arith.constant 0xFF80 : bf16
    %init_max = linalg.fill ins(%cst_init_max : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %cst_init_sum = arith.constant -0.000000e+00 : bf16
    %init_sum = linalg.fill ins(%cst_init_sum : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#x]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %xm = linalg.generic {indexing_maps = [#tile, #tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%x, %hi1 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %cl: bf16, %out: bf16):
      %m = arith.minnumf %in, %cl : bf16
      linalg.yield %m : bf16
    } -> tensor<32x32xbf16>
    %mx = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%xm : tensor<32x32xbf16>) outs(%init_max : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>
    %bm = linalg.broadcast ins(%mx : tensor<32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) dimensions = [1]
    %d = linalg.sub ins(%x, %bm : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %e = linalg.exp ins(%d : tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %em1 = linalg.generic {indexing_maps = [#tile, #tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%e, %hi2 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %cl: bf16, %out: bf16):
      %m = arith.minnumf %in, %cl : bf16
      linalg.yield %m : bf16
    } -> tensor<32x32xbf16>
    %em = linalg.generic {indexing_maps = [#tile, #tile, #tile], iterator_types = ["parallel", "parallel"]} ins(%em1, %lo2 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty_tile : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %cl: bf16, %out: bf16):
      %m = arith.maxnumf %in, %cl : bf16
      linalg.yield %m : bf16
    } -> tensor<32x32xbf16>
    %sm = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%em : tensor<32x32xbf16>) outs(%init_sum : tensor<32xbf16>) {
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
