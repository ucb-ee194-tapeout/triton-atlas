// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt.dot other than FP8 E4M3 x FP8 E4M3 into f32: Atlas's MXU takes FP8 tiles, and pass 5 expects Triton's f32 dot
// result (its type labels the MXU's BF16 accumulator). BF16 dot inputs, the common Inductor case, need an FP8
// conversion and a scale-code rule that don't exist yet.

tt.func @bf16(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  // expected-error @below {{atlas: unsupported tt.dot: multiplies bf16; Atlas's MXU takes FP8 E4M3 only}}
  // expected-error @below {{failed to legalize operation 'tt.dot' that was explicitly marked illegal}}
  %d = tt.dot %a, %b, %zero : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @bf16_acc(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
  // expected-error @below {{atlas: unsupported tt.dot: accumulates in bf16; pass 5 expects Triton's f32 dot result}}
  // expected-error @below {{failed to legalize operation 'tt.dot' that was explicitly marked illegal}}
  %d = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xbf16>
  tt.return
}
