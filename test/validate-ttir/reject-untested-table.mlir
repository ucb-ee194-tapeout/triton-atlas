// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tanh, sin, cos, log2, log: Atlas's tables are untested on hardware.

tt.func @tanh(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported math.tanh: no lowering yet: Atlas's table for this function is untested on hardware}}
  // expected-error @below {{failed to legalize operation 'math.tanh' that was explicitly marked illegal}}
  %r = math.tanh %x : tensor<32x32xf32>
  tt.return
}
