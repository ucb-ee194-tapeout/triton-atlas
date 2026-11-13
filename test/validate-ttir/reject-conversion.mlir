// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Conversions Atlas has no op for.

tt.func @sitofp(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported arith.sitofp: no direct Atlas equivalent: a conversion between integer and float, to other float formats, scaled, or flushing denormals}}
  // expected-error @below {{failed to legalize operation 'arith.sitofp' that was explicitly marked illegal}}
  %r = arith.sitofp %i : tensor<32x32xi32> to tensor<32x32xf32>
  tt.return
}

// -----

// tt.fp_to_fp narrowing f32 to BF16 must round to nearest even, as Atlas does.
tt.func @fp_to_fp_rtz(%x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported tt.fp_to_fp: narrows with rounding rtz; Atlas rounds to nearest even (rtne) only}}
  // expected-error @below {{failed to legalize operation 'tt.fp_to_fp' that was explicitly marked illegal}}
  %r = tt.fp_to_fp %x, rounding = rtz : tensor<32x32xf32> -> tensor<32x32xbf16>
  tt.return
}
