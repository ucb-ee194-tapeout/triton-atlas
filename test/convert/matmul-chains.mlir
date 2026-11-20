// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// A 64x64 matmul as 2x2 output tiles over a K of two tiles, the form pass 4 leaves: the accumulator grid comes from a
// pack of an empty tensor, each tile is filled with zero, and every output tile is a chain of two matmuls through
// the accumulator. Slices and inserts only move tiles around. Each chain is one contiguous group (weight load, reset,
// weight load, accumulate, readout) on MXU0, emitted where the chain ends, with its inputs read just before it. A
// slot read by two chains (input 0, input 4, ...) is read once.

// CHECK-LABEL: func.func @matmul_64() -> !atlas.virtual_state
// CHECK-SAME: atlas.output_dram_base = 2415935488 : i64
// CHECK: "atlas.virtual_input_fp8"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: "atlas.virtual_input_fp8"(%{{.*}}) <{index = 4 : i32}>
// CHECK-NEXT: "atlas.virtual_input_fp8"(%{{.*}}) <{index = 1 : i32}>
// CHECK-NEXT: "atlas.virtual_input_fp8"(%{{.*}}) <{index = 6 : i32}>
// CHECK-NEXT: "atlas.virtual_mxu_load_weight"
// CHECK-NEXT: "atlas.virtual_mxu_reset"
// CHECK-NEXT: "atlas.virtual_mxu_load_weight"
// CHECK-NEXT: "atlas.virtual_mxu_accumulate"
// CHECK-NEXT: "atlas.virtual_mxu_readout_bf16"
// CHECK-NEXT: "atlas.virtual_input_fp8"(%{{.*}}) <{index = 5 : i32}>
// CHECK-NEXT: "atlas.virtual_input_fp8"(%{{.*}}) <{index = 7 : i32}>
// CHECK-NEXT: "atlas.virtual_mxu_load_weight"
// CHECK-NEXT: "atlas.virtual_mxu_reset"
// CHECK-NEXT: "atlas.virtual_mxu_load_weight"
// CHECK-NEXT: "atlas.virtual_mxu_accumulate"
// CHECK-NEXT: "atlas.virtual_mxu_readout_bf16"
// CHECK: "atlas.virtual_output_bf16"({{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: "atlas.virtual_output_bf16"({{.*}}) <{index = 1 : i32}>
// CHECK-NEXT: "atlas.virtual_output_bf16"({{.*}}) <{index = 2 : i32}>
// CHECK-NEXT: "atlas.virtual_output_bf16"({{.*}}) <{index = 3 : i32}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#a0 = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#a1 = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = activation_fp8, offset = 32, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#a2 = #ttatlas.slot<index = 2, kind = memory, arg = 0, role = activation_fp8, offset = 2048, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#a3 = #ttatlas.slot<index = 3, kind = memory, arg = 0, role = activation_fp8, offset = 2080, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#b0 = #ttatlas.slot<index = 4, kind = memory, arg = 1, role = weight_fp8_nk, offset = 0, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#b1 = #ttatlas.slot<index = 5, kind = memory, arg = 1, role = weight_fp8_nk, offset = 32, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#b2 = #ttatlas.slot<index = 6, kind = memory, arg = 1, role = weight_fp8_nk, offset = 2048, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#b3 = #ttatlas.slot<index = 7, kind = memory, arg = 1, role = weight_fp8_nk, offset = 2080, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
#o0 = #ttatlas.slot<index = 0, kind = memory, arg = 2, role = output_bf16, offset = 0, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#o1 = #ttatlas.slot<index = 1, kind = memory, arg = 2, role = output_bf16, offset = 32, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#o2 = #ttatlas.slot<index = 2, kind = memory, arg = 2, role = output_bf16, offset = 2048, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#o3 = #ttatlas.slot<index = 3, kind = memory, arg = 2, role = output_bf16, offset = 2080, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @matmul_64(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<bf16>) {
    %zero = arith.constant 0.000000e+00 : bf16
    %pa_ptrs = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %pb_ptrs = tt.splat %b : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %pc = tt.splat %c : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %xa = tt.load %pa_ptrs : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %xb = tt.load %pb_ptrs : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %acc_host = tensor.empty() : tensor<64x64xbf16>
    %acc_grid = tensor.empty() : tensor<2x2x32x32xbf16>
    %grid = linalg.pack %acc_host inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %acc_grid : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
    %operand_grid = tensor.empty() : tensor<2x2x32x32xf8E4M3FN>
    %pa = linalg.pack %xa inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %operand_grid {ttatlas.slot = [#a0, #a1, #a2, #a3]} : tensor<64x64xf8E4M3FN> -> tensor<2x2x32x32xf8E4M3FN>
    %pb = linalg.pack %xb inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %operand_grid {ttatlas.slot = [#b0, #b1, #b2, #b3]} : tensor<64x64xf8E4M3FN> -> tensor<2x2x32x32xf8E4M3FN>
    %c00 = tensor.extract_slice %grid[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %z00 = linalg.fill ins(%zero : bf16) outs(%c00 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %c01 = tensor.extract_slice %grid[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %z01 = linalg.fill ins(%zero : bf16) outs(%c01 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %c10 = tensor.extract_slice %grid[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %z10 = linalg.fill ins(%zero : bf16) outs(%c10 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %c11 = tensor.extract_slice %grid[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %z11 = linalg.fill ins(%zero : bf16) outs(%c11 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %a00_0 = tensor.extract_slice %pa[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b00_0 = tensor.extract_slice %pb[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m000 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a00_0, %b00_0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%z00 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %a01_0 = tensor.extract_slice %pa[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b10_0 = tensor.extract_slice %pb[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m001 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a01_0, %b10_0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m000 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %r00 = tensor.insert_slice %m001 into %grid[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    %a00_1 = tensor.extract_slice %pa[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b01_0 = tensor.extract_slice %pb[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m010 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a00_1, %b01_0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%z01 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %a01_1 = tensor.extract_slice %pa[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b11_0 = tensor.extract_slice %pb[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m011 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a01_1, %b11_0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m010 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %r01 = tensor.insert_slice %m011 into %r00[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    %a10_0 = tensor.extract_slice %pa[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b00_1 = tensor.extract_slice %pb[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m100 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a10_0, %b00_1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%z10 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %a11_0 = tensor.extract_slice %pa[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b10_1 = tensor.extract_slice %pb[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m101 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a11_0, %b10_1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m100 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %r10 = tensor.insert_slice %m101 into %r01[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    %a10_1 = tensor.extract_slice %pa[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b01_1 = tensor.extract_slice %pb[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m110 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a10_1, %b01_1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%z11 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %a11_1 = tensor.extract_slice %pa[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %b11_1 = tensor.extract_slice %pb[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xf8E4M3FN> to tensor<32x32xf8E4M3FN>
    %m111 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%a11_1, %b11_1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m110 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %r11 = tensor.insert_slice %m111 into %r10[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    %host = tensor.empty() : tensor<64x64xbf16>
    %out = linalg.unpack %r11 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %host {ttatlas.slot = [#o0, #o1, #o2, #o3]} : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
    tt.store %pc, %out : tensor<64x64x!tt.ptr<bf16>>
    tt.return
  }
}
