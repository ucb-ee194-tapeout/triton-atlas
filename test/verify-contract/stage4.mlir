// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=4

// Pass 4's contract: tt.dot operands are FP8 E4M3; every other float value is BF16 or FP8 (dots in a
// ttatlas.k_chain are exempt, since their accumulator stays in the MXU).

tt.func @ok(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
  %d = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xbf16>
  %r = arith.maximumf %d, %zero : tensor<32x32xbf16>
  tt.return
}

// -----

tt.func @f32_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 4 broken: float value or pointer is neither BF16 nor FP8}}
  %y = arith.extf %x : tensor<32x32xbf16> to tensor<32x32xf32>
  tt.return
}

// -----

tt.func @bf16_dot(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
  // expected-error @below {{contract of pass 4 broken: tt.dot operands are not FP8 E4M3}}
  %d = tt.dot %a, %b, %zero : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xbf16>
  tt.return
}

// -----

// A K chain keeps its accumulator in the MXU: its f32 zero seed and chained results are allowed.
tt.func @chain_ok(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %d1 = tt.dot %a, %b, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %out = arith.truncf %d1 : tensor<32x32xf32> to tensor<32x32xbf16>
  tt.return
}

// -----

// Output pointers are retyped too: an f32 pointer is left.
// expected-error @below {{contract of pass 4 broken: float value or pointer is neither BF16 nor FP8}}
tt.func @f32_pointer(%p: !tt.ptr<f32>) {
  tt.return
}

// -----

// The pointee is checked inside tensors of pointers too.
tt.func @f32_pointer_tensor(%x: tensor<32xi64>) {
  // expected-error @below {{contract of pass 4 broken: float value or pointer is neither BF16 nor FP8}}
  %s = tt.int_to_ptr %x : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
  tt.return
}
