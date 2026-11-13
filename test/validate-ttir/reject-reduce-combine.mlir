// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// A reduction whose combine isn't one add, max or min: the VPU's row reductions have only those.

tt.func @product(%x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported tt.reduce: the combine is not one float add, max or min of one tensor}}
  // expected-error @below {{failed to legalize operation 'tt.reduce' that was explicitly marked illegal}}
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.mulf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<32x32xf32>) -> tensor<32xf32>
  tt.return
}
