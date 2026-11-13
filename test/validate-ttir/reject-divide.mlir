// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// arith.divf other than 1 / x: Atlas has only a reciprocal table.

tt.func @x_over_y(%x: tensor<32x32xf32>, %y: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported arith.divf: only 1 / x is accepted (Atlas's reciprocal table); x / y needs a rewrite that doesn't exist yet}}
  // expected-error @below {{failed to legalize operation 'arith.divf' that was explicitly marked illegal}}
  %r = arith.divf %x, %y : tensor<32x32xf32>
  tt.return
}
