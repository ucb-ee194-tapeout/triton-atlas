// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Conversions Atlas has no op for.

tt.func @sitofp(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported arith.sitofp: no direct Atlas equivalent: a conversion between integer and float, to other float formats, scaled, or flushing denormals}}
  // expected-error @below {{failed to legalize operation 'arith.sitofp' that was explicitly marked illegal}}
  %r = arith.sitofp %i : tensor<32x32xi32> to tensor<32x32xf32>
  tt.return
}
