// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --verify-diagnostics

// An f32 source is rejected under the precision policy fp8-bf16-v1, the default when the module has no
// ttatlas.precision_policy tag: converting an f32 kernel changes its results, so f32 sources wait for a policy whose
// error is measured.

tt.func @f32_source(%p: !tt.ptr<f32>) {
  %s = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  // expected-error @below {{atlas: unsupported tt.load: an f32 source; the precision policy fp8-bf16-v1 takes FP8 and BF16 sources only}}
  %v = tt.load %s : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

// The policy is read from the module tag when present.
module attributes {ttatlas.precision_policy = "fp8-bf16-v1"} {
  tt.func @f32_source_tagged(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    // expected-error @below {{atlas: unsupported tt.load: an f32 source; the precision policy fp8-bf16-v1 takes FP8 and BF16 sources only}}
    %v = tt.load %s : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}
