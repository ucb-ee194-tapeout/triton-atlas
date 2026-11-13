// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt.trans other than on an FP8 dot's second operand (the weight): Atlas transposes FP8 matmul weights only.

tt.func @bf16(%x: tensor<32x32xbf16>) {
  // expected-error @below {{atlas: unsupported tt.trans: transposes bf16; Atlas transposes FP8 only}}
  // expected-error @below {{failed to legalize operation 'tt.trans' that was explicitly marked illegal}}
  %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
  tt.return
}

// -----

tt.func @fp8_not_dot(%x: tensor<32x32xf8E4M3FN>) {
  // expected-error @below {{atlas: unsupported tt.trans: the transpose feeds something other than a tt.dot's second operand; Atlas transposes FP8 weights only}}
  // expected-error @below {{failed to legalize operation 'tt.trans' that was explicitly marked illegal}}
  %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
  %e = arith.addf %t, %t : tensor<32x32xf8E4M3FN>
  tt.return
}

// -----

tt.func @fp8_lhs(%x: tensor<32x32xf8E4M3FN>, %w: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  // expected-error @below {{atlas: unsupported tt.trans: the transpose feeds something other than a tt.dot's second operand; Atlas transposes FP8 weights only}}
  // expected-error @below {{failed to legalize operation 'tt.trans' that was explicitly marked illegal}}
  %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
  %d = tt.dot %t, %w, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}
