// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// An f32 source: Atlas has no f32 datapath, and no precision policy allows one yet. f32 outputs are allowed.

tt.func @f32_load(%p: !tt.ptr<f32>) {
  %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  // expected-error @below {{atlas: unsupported tt.load: an f32 source; Atlas has no f32 datapath and no precision policy allows one yet}}
  // expected-error @below {{failed to legalize operation 'tt.load' that was explicitly marked illegal}}
  %v = tt.load %ps : tensor<32x32x!tt.ptr<f32>>
  tt.return
}
