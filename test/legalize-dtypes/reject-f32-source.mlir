// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --verify-diagnostics

// An f32 source is rejected: converting an f32 kernel changes its results, so f32 sources wait until the error
// of doing so is measured.

tt.func @f32_source(%p: !tt.ptr<f32>) {
  %s = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  // expected-error @below {{atlas: unsupported tt.load: f32 inputs are not supported yet; use fp8 or bf16}}
  %v = tt.load %s : tensor<32x32x!tt.ptr<f32>>
  tt.return
}
