// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Conversions other than BF16 <-> f32, such as to FP8 inside the kernel: no rule picks the scale code yet.

tt.func @truncf_to_fp8(%x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported arith.truncf: converts f32 to f8E4M3FN; only BF16 <-> f32 is accepted (no FP8 scale-code rule yet)}}
  // expected-error @below {{failed to legalize operation 'arith.truncf' that was explicitly marked illegal}}
  %r = arith.truncf %x : tensor<32x32xf32> to tensor<32x32xf8E4M3FN>
  tt.return
}

// -----

tt.func @fp_to_fp_to_fp8(%x: tensor<32x32xbf16>) {
  // expected-error @below {{atlas: unsupported tt.fp_to_fp: converts bf16 to f8E4M3FN; only BF16 <-> f32 is accepted (no FP8 scale-code rule yet)}}
  // expected-error @below {{failed to legalize operation 'tt.fp_to_fp' that was explicitly marked illegal}}
  %r = tt.fp_to_fp %x, rounding = rtne : tensor<32x32xbf16> -> tensor<32x32xf8E4M3FN>
  tt.return
}
