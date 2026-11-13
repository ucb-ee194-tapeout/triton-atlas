// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Ops with no Atlas meaning: inline assembly, console, calls, pointer casts, grid dependencies.

tt.func @print(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.print: no Atlas meaning: inline assembly, console, calls after inlining, pointer casts, grid dependencies}}
  // expected-error @below {{failed to legalize operation 'tt.print' that was explicitly marked illegal}}
  tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %x : tensor<32x32xf32>
  tt.return
}
