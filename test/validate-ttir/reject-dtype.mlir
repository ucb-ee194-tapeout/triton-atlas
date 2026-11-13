// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Other dtypes the precision policy doesn't take: Atlas loads FP8 E4M3 or BF16 and writes BF16, which the driver
// converts to f32 if the output is f32.

tt.func @f16_load(%p: !tt.ptr<f16>) {
  %ps = tt.splat %p : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
  // expected-error @below {{atlas: unsupported tt.load: loads f16; Atlas loads FP8 E4M3 or BF16 only}}
  // expected-error @below {{failed to legalize operation 'tt.load' that was explicitly marked illegal}}
  %v = tt.load %ps : tensor<32x32x!tt.ptr<f16>>
  tt.return
}

// -----

tt.func @fp8_store(%p: !tt.ptr<f8E4M3FN>, %v: tensor<32x32xf8E4M3FN>) {
  %ps = tt.splat %p : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  // expected-error @below {{atlas: unsupported tt.store: stores f8E4M3FN; Atlas writes BF16, which the driver converts to f32 only}}
  // expected-error @below {{failed to legalize operation 'tt.store' that was explicitly marked illegal}}
  tt.store %ps, %v : tensor<32x32x!tt.ptr<f8E4M3FN>>
  tt.return
}
