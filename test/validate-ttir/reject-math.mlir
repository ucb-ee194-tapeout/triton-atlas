// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Other math functions: no Atlas op.

tt.func @absf(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported math.absf: no direct Atlas equivalent: no op for this function}}
  // expected-error @below {{failed to legalize operation 'math.absf' that was explicitly marked illegal}}
  %r = math.absf %x : tensor<32x32xf32>
  tt.return
}
