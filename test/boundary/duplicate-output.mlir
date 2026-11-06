// RUN: not triton-atlas-opt %s --convert-triton-to-atlas-virtual 2>&1 | FileCheck %s
// CHECK: outputs require unmasked BF16 tiles with unique output_bf16 slots
module {
  tt.func @copy(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %v = tt.load %p {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = constant_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %q, %v {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f32>} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %q, %v {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f32>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
