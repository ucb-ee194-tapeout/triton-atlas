// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Integer division and remainder.

tt.func @divsi(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported arith.divsi: no direct Atlas equivalent: integer division and remainder}}
  // expected-error @below {{failed to legalize operation 'arith.divsi' that was explicitly marked illegal}}
  %r = arith.divsi %i, %i : tensor<32x32xi32>
  tt.return
}
