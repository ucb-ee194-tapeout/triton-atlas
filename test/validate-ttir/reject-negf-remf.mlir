// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// arith.negf and arith.remf: no Atlas op.

tt.func @negf(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported arith.negf: no direct Atlas equivalent: no op for this (negf could be built as x * -1)}}
  // expected-error @below {{failed to legalize operation 'arith.negf' that was explicitly marked illegal}}
  %r = arith.negf %x : tensor<32x32xf32>
  tt.return
}
