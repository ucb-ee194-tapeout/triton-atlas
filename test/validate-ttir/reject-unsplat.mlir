// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt.unsplat: scalar data has no Atlas path.

tt.func @unsplat(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  %ps = tt.splat %p : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  // expected-error @below {{atlas: unsupported tt.unsplat: no lowering yet: scalar data has no Atlas path in virtual IR}}
  // expected-error @below {{failed to legalize operation 'tt.unsplat' that was explicitly marked illegal}}
  %u = tt.unsplat %ps : tensor<1x!tt.ptr<f32>>
  tt.return
}
