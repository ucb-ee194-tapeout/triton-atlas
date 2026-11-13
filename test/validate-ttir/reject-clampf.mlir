// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt.clampf: no pattern yet.

tt.func @clampf(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.clampf: no direct Atlas equivalent: no clamp op (could be built from a max then a min)}}
  // expected-error @below {{failed to legalize operation 'tt.clampf' that was explicitly marked illegal}}
  %r = tt.clampf %x, %x, %x, propagateNan = none : tensor<32x32xf32>
  tt.return
}
