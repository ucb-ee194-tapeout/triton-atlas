// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// An op from outside TTIR's dialects.

tt.func @not_ttir(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tensor.empty: not an op TTIR contains (not in pass 1's op inventory)}}
  // expected-error @below {{failed to legalize operation 'tensor.empty' that was explicitly marked illegal}}
  %r = tensor.empty() : tensor<32x32xf32>
  tt.return
}
