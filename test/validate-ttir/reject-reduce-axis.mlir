// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Reductions other than along the rows of a 2-D tile: the column reduce isn't in virtual IR.

tt.func @axis0(%x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported tt.reduce: reduces axis 0 of a 2-D tensor; Atlas reduces rows of a 2-D tile only (axis 1)}}
  // expected-error @below {{failed to legalize operation 'tt.reduce' that was explicitly marked illegal}}
  %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<32x32xf32>) -> tensor<32xf32>
  tt.return
}
