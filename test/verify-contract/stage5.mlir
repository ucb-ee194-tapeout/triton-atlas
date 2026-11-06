// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=5

// Pass 5's contract: linalg.matmul inputs are FP8 E4M3; every other float value or pointer is BF16 or FP8 (a matmul
// chain's accumulator, the matmuls and their linalg.fill seed, may be f32, since it stays in the MXU).

#identity = affine_map<(d0, d1) -> (d0, d1)>

// A two-step K chain into an f32 accumulator, read out as BF16.
tt.func @chain_ok(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m0 = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m1 = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %out = tensor.empty() : tensor<32x32xbf16>
  %r = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]} ins(%m1 : tensor<32x32xf32>) outs(%out : tensor<32x32xbf16>) {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.return
}

// -----

tt.func @bf16_matmul(%a: tensor<32x32xbf16>) {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  // expected-error @below {{contract of pass 5 broken: linalg.matmul inputs are not FP8 E4M3}}
  %m = linalg.matmul ins(%a, %a : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}

// -----

// f32 outside a matmul chain is left.
tt.func @f32_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 5 broken: float value or pointer is neither BF16 nor FP8}}
  %e = tensor.empty() : tensor<32x32xf32>
  tt.return
}

// -----

// Output pointers are retyped too: an f32 pointer is left.
// expected-error @below {{contract of pass 5 broken: float value or pointer is neither BF16 nor FP8}}
tt.func @f32_pointer(%p: !tt.ptr<f32>) {
  tt.return
}

// -----

// The pointee is checked inside tensors of pointers too.
tt.func @f32_pointer_tensor(%x: tensor<32xi64>) {
  // expected-error @below {{contract of pass 5 broken: float value or pointer is neither BF16 nor FP8}}
  %s = tt.int_to_ptr %x : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
  tt.return
}
