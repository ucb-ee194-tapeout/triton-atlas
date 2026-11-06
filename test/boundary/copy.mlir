// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.'
// CHECK: func.func @copy() -> !atlas.virtual_state
// CHECK-SAME: atlas.input_dram_base = 2415919104 : i64
// CHECK-SAME: atlas.output_dram_base = 2415921152 : i64
// CHECK: %[[S:.*]] = "atlas.virtual_start"
// CHECK: %[[N:.*]], %[[V:.*]] = "atlas.virtual_input_bf16"(%[[S]])
// CHECK-SAME: index = 0 : i32
// CHECK: %[[O:.*]] = "atlas.virtual_output_bf16"(%[[N]], %[[V]])
// CHECK-SAME: index = 0 : i32
// CHECK: return %[[O]] : !atlas.virtual_state
module {
  tt.func @copy(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %v = tt.load %p {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = constant_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %q, %v {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f32>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
