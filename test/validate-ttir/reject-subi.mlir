// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// arith.subi: pass 6, which proves each address is one whole tile, doesn't evaluate subtraction yet. When it does,
// this row becomes legal in the same change.

tt.func @subi(%i: tensor<32x32xi32>) {
  // expected-error @below {{atlas: unsupported arith.subi: no lowering yet: subtraction in address math; pass 6 doesn't evaluate it yet}}
  // expected-error @below {{failed to legalize operation 'arith.subi' that was explicitly marked illegal}}
  %r = arith.subi %i, %i : tensor<32x32xi32>
  tt.return
}
