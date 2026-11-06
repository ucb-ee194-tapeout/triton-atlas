// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s
// CHECK: atlas.output_dram_base = 2415923200 : i64
// CHECK: atlas.virtual_input_bf16
// CHECK-SAME: index = 0 : i32
// CHECK: atlas.virtual_input_fp8
// CHECK-SAME: index = 1 : i32
// CHECK: atlas.virtual_output_bf16
module {
  tt.func @copy(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %w: !tt.ptr<f8E4M3FN>) {
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %wp = tt.splat %w : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %wv = tt.load %wp {ttatlas.slot = #ttatlas.slot<index = 1, arg = 2, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %v = tt.load %p {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = constant_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %q, %v {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f32>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
