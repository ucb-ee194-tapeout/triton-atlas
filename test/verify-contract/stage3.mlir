// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=3

// Pass 3's contract: every 2-D value is 32x32; every tt.dot carries ttatlas.mxu.

tt.func @ok(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d = tt.dot %a, %b, %zero {ttatlas.mxu = 0 : i32} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @big_block() {
  // expected-error @below {{contract of pass 3 broken: 2-D value is not 32x32}}
  %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  tt.return
}

// -----

tt.func @no_unit(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  // expected-error @below {{contract of pass 3 broken: tt.dot without ttatlas.mxu}}
  %d = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

// Block arguments count too.
// expected-error @below {{contract of pass 3 broken: 2-D value is not 32x32}}
tt.func @big_arg(%x: tensor<64x64xbf16>) {
  tt.return
}
