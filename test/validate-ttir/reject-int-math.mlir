// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Integer ops other than the address math addi and muli.

tt.func @shli(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported arith.shli: no direct Atlas equivalent: an integer operation (address math uses addi and muli)}}
  // expected-error @below {{failed to legalize operation 'arith.shli' that was explicitly marked illegal}}
  %r = arith.shli %i, %i : tensor<32x32xi32>
  tt.return
}
