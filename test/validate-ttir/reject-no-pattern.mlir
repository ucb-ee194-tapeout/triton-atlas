// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Ops with no Atlas pattern yet (dot_scaled, reshape, cat, join, split, scan, map_elementwise).

tt.func @reshape(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.reshape: no lowering yet: no Atlas pattern for this op}}
  // expected-error @below {{failed to legalize operation 'tt.reshape' that was explicitly marked illegal}}
  %r = tt.reshape %x : tensor<32x32xf32> -> tensor<1024xf32>
  tt.return
}
