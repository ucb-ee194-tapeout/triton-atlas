// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt divide and square root ops: Atlas has no divide or square root op.

tt.func @precise_divf(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.precise_divf: no direct Atlas equivalent: no divide op (could be built from 1 / y through the reciprocal table, then a multiply)}}
  // expected-error @below {{failed to legalize operation 'tt.precise_divf' that was explicitly marked illegal}}
  %r = tt.precise_divf %x, %x : tensor<32x32xf32>
  tt.return
}
