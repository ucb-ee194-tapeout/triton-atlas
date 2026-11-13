// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual | FileCheck %s

// A kernel with compute between its load and store is not converted yet (no compute patterns): pass 7 leaves the whole
// function as it is, without an error, instead of converting the boundary around TTIR it cannot lower. copy.mlir is the
// same kernel without the add, which converts.
// CHECK: tt.func @add
// CHECK: linalg.add
// CHECK-NOT: atlas.virtual
module {
  tt.func @add(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %v = tt.load %p {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = constant_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %e = tensor.empty() : tensor<32x32xbf16>
    %s = linalg.add ins(%v, %v : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    tt.store %q, %s {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f32>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
