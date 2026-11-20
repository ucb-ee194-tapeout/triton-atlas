// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// A row max over a 32x64 row of two tiles, twice (64x64): each tile's reduction passes the earlier partial result through
// its init, which pass 7 turns into a reduction of the tile and a max with the partial. The first tile's init is an
// identity fill and costs nothing. The results are vector tiles (one value per row) stored to rank-one slots.

// CHECK-LABEL: func.func @row_max_64()
// CHECK: %[[T00:.*]] = "atlas.virtual_vpu_reduce"({{.*}}) <{kind = "row_max"}>
// CHECK: %[[T01:.*]] = "atlas.virtual_vpu_reduce"({{.*}}) <{kind = "row_max"}>
// CHECK-NEXT: %[[R0:.*]] = "atlas.virtual_vpu_binary"(%[[T01]], %[[T00]]) <{kind = "max"}>
// CHECK: %[[T10:.*]] = "atlas.virtual_vpu_reduce"({{.*}}) <{kind = "row_max"}>
// CHECK: %[[T11:.*]] = "atlas.virtual_vpu_reduce"({{.*}}) <{kind = "row_max"}>
// CHECK-NEXT: %[[R1:.*]] = "atlas.virtual_vpu_binary"(%[[T11]], %[[T10]]) <{kind = "max"}>
// CHECK: "atlas.virtual_output_bf16"({{.*}}, %[[R0]]) <{index = 0 : i32}>
// CHECK-NEXT: "atlas.virtual_output_bf16"({{.*}}, %[[R1]]) <{index = 1 : i32}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#tile = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0) -> (d0)>
#rowmap = affine_map<(d0, d1) -> (d0)>
#x0 = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#x1 = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = constant_bf16, offset = 32, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#x2 = #ttatlas.slot<index = 2, kind = memory, arg = 0, role = constant_bf16, offset = 2048, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#x3 = #ttatlas.slot<index = 3, kind = memory, arg = 0, role = constant_bf16, offset = 2080, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y0 = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = rows, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#y1 = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = output_bf16, offset = 32, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = rows, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @row_max_64(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %empty_tile = tensor.empty() : tensor<32x32xbf16>
    %empty_vec = tensor.empty() : tensor<32xbf16>
    %cst_init_max = arith.constant 0xFF80 : bf16
    %init_max = linalg.fill ins(%cst_init_max : bf16) outs(%empty_vec : tensor<32xbf16>) -> tensor<32xbf16>
    %pp = tt.splat %a : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %lx = tt.load %pp : tensor<64x64x!tt.ptr<bf16>>
    %empty_grid = tensor.empty() : tensor<2x2x32x32xbf16>
    %px = linalg.pack %lx inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %empty_grid {ttatlas.slot = [#x0, #x1, #x2, #x3]} : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
    %empty_rows = tensor.empty() : tensor<2x32xbf16>
    %t00 = tensor.extract_slice %px[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %t01 = tensor.extract_slice %px[0, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %r00 = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%t00 : tensor<32x32xbf16>) outs(%init_max : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>

    %r01 = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%t01 : tensor<32x32xbf16>) outs(%r00 : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>

    %v0 = tensor.insert_slice %r01 into %empty_rows[0, 0] [1, 32] [1, 1] : tensor<32xbf16> into tensor<2x32xbf16>
    %t10 = tensor.extract_slice %px[1, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %t11 = tensor.extract_slice %px[1, 1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
    %r10 = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%t10 : tensor<32x32xbf16>) outs(%init_max : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>

    %r11 = linalg.generic {indexing_maps = [#tile, #rowmap], iterator_types = ["parallel", "reduction"]} ins(%t11 : tensor<32x32xbf16>) outs(%r10 : tensor<32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %res = arith.maxnumf %out, %in : bf16
      linalg.yield %res : bf16
    } -> tensor<32xbf16>

    %v1 = tensor.insert_slice %r11 into %v0[1, 0] [1, 32] [1, 1] : tensor<32xbf16> into tensor<2x32xbf16>
    %host = tensor.empty() : tensor<64xbf16>
    %out = linalg.unpack %v1 inner_dims_pos = [0] inner_tiles = [32] into %host {ttatlas.slot = [#y0, #y1]} : tensor<2x32xbf16> -> tensor<64xbf16>
    %pq = tt.splat %b : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    tt.store %pq, %out : tensor<64x!tt.ptr<bf16>>
    tt.return
  }
}
