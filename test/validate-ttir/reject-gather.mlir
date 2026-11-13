// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Gathers and scatters: data-dependent addresses.

tt.func @gather(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.gather: no direct Atlas equivalent: data-dependent addresses; atlas-mlir accepts only constant whole-tile transfers}}
  // expected-error @below {{failed to legalize operation 'tt.gather' that was explicitly marked illegal}}
  %g = tt.gather %x[%i] {axis = 1 : i32} : (tensor<32x32xf32>, tensor<32x32xi32>) -> tensor<32x32xf32>
  tt.return
}
