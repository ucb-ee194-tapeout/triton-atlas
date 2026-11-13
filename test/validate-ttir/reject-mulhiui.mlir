// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt.mulhiui: Atlas has no op for this integer operation (address math uses addi and muli).

tt.func @mulhiui(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.mulhiui: no direct Atlas equivalent: an integer operation (address math uses addi and muli)}}
  // expected-error @below {{failed to legalize operation 'tt.mulhiui' that was explicitly marked illegal}}
  %r = tt.mulhiui %i, %i : tensor<32x32xi32>
  tt.return
}
