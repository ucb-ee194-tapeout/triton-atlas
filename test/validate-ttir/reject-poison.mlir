// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// A used ub.poison: an undefined value has no Atlas meaning (an unused one is accepted).

tt.func @used(%x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported ub.poison: a used poison value has no Atlas meaning}}
  // expected-error @below {{failed to legalize operation 'ub.poison' that was explicitly marked illegal}}
  %p = ub.poison : tensor<32x32xf32>
  %r = arith.addf %x, %p : tensor<32x32xf32>
  tt.return
}
