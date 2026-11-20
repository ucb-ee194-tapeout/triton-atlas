// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not=ttatlas.slot

// A kernel with no compute has its slots on the tt.load and tt.store themselves (no pack or unpack exists): input,
// then a move, then output; the f32 host cast and the masked lanes (other = -inf) belong to the slots, not the IR.
// Where an input reaches an output through a pack and an unpack, the same move keeps an input's register out of the
// DMA store.

// CHECK-LABEL: func.func @copy_direct()
// CHECK-SAME: \22other\22:65408
// CHECK-SAME: \22host_dtype\22:\22f32\22
// CHECK: %[[S1:.*]], %[[V:.*]] = "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: %[[M:.*]] = "atlas.virtual_vpu_unary"(%[[V]]) <{kind = "mov"}>
// CHECK-NEXT: %[[S2:.*]] = "atlas.virtual_output_bf16"(%[[S1]], %[[M]]) <{index = 0 : i32}>
// CHECK-NEXT: return %[[S2]]

// CHECK-LABEL: func.func @copy_packed()
// CHECK: "atlas.virtual_input_bf16"(%{{.*}}) <{index = 0 : i32}>
// CHECK-NEXT: "atlas.virtual_vpu_unary"({{.*}}) <{kind = "mov"}>
// CHECK-NEXT: "atlas.virtual_output_bf16"
#i = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20], other = -inf : bf16, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#o = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20], vector_layout = none, transposed = false, host_dtype = f32, tile_dtype = bf16>
#y = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module {
  tt.func @copy_direct(%a: !tt.ptr<bf16>, %b: !tt.ptr<f32>) {
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %v = tt.load %p {ttatlas.slot = #i} : tensor<32x32x!tt.ptr<bf16>>
    %w = arith.extf %v : tensor<32x32xbf16> to tensor<32x32xf32>
    tt.store %q, %w {ttatlas.slot = #o} : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
  tt.func @copy_packed(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %p_x = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %l_x = tt.load %p_x : tensor<32x32x!tt.ptr<bf16>>
    %g_x = tensor.empty() : tensor<1x1x32x32xbf16>
    %k_x = linalg.pack %l_x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %g_x {ttatlas.slot = [#i]} : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %x = tensor.collapse_shape %k_x [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
    %e_y = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
    %h_y = tensor.empty() : tensor<32x32xbf16>
    %u_y = linalg.unpack %e_y inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %h_y {ttatlas.slot = [#y]} : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
    %q_y = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    tt.store %q_y, %u_y : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
