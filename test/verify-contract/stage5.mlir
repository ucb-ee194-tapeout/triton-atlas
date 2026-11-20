// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=5

// Pass 5's contract: linalg.matmul inputs are FP8 E4M3; every other float value is BF16 or FP8, accumulators included,
// inside linalg bodies too. The exception is the extf or truncf that casts a value to the host type of the store
// reading it: a store keeps the host's pointer type.

#identity = affine_map<(d0, d1) -> (d0, d1)>

// A two-step K chain with a BF16 accumulator.
tt.func @chain_ok(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<32x32xbf16>
  %acc = linalg.fill ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %m1 = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m0 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  tt.return
}

// -----

// One cast that two stores share (CSE merges equal casts) is still the stores' cast.
tt.func @cast_shared_by_stores(%p: tensor<32x32x!tt.ptr<f32>>, %q: tensor<32x32x!tt.ptr<f32>>, %x: tensor<32x32xbf16>) {
  %w = arith.extf %x : tensor<32x32xbf16> to tensor<32x32xf32>
  tt.store %p, %w : tensor<32x32x!tt.ptr<f32>>
  tt.store %q, %w : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @bf16_matmul(%a: tensor<32x32xbf16>) {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<32x32xbf16>
  %acc = linalg.fill ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // expected-error @below {{contract of pass 5 broken: linalg.matmul inputs are not FP8 E4M3}}
  %m = linalg.matmul ins(%a, %a : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%acc : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  tt.return
}

// -----

// f32 outside a matmul chain is left.
tt.func @f32_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 5 broken: float value is neither BF16 nor FP8}}
  %e = tensor.empty() : tensor<32x32xf32>
  tt.return
}

// -----

// Loads and stores keep the host's pointer types, f32 pointers included.
tt.func @f32_pointer(%p: !tt.ptr<f32>, %x: tensor<32xi64>) {
  %s = tt.int_to_ptr %x : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
  tt.return
}

// -----

// A BF16 value stored to an f32 pointer goes through an extf read only by the store.
tt.func @store_cast(%p: tensor<32x32x!tt.ptr<f32>>, %x: tensor<32x32xbf16>) {
  %w = arith.extf %x : tensor<32x32xbf16> to tensor<32x32xf32>
  tt.store %p, %w : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

// The same extf also read by compute is f32 compute.
tt.func @cast_also_computed(%p: tensor<32x32x!tt.ptr<f32>>, %x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 5 broken: float value is neither BF16 nor FP8}}
  %w = arith.extf %x : tensor<32x32xbf16> to tensor<32x32xf32>
  // expected-error @below {{contract of pass 5 broken: float value is neither BF16 nor FP8}}
  %t = arith.addf %w, %w : tensor<32x32xf32>
  tt.store %p, %w : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

// f32 math inside a linalg body is f32 compute too, though the generic's result is BF16.
tt.func @f32_in_body(%x: tensor<32x32xbf16>) {
  %out = tensor.empty() : tensor<32x32xbf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%out : tensor<32x32xbf16>) {
  ^bb0(%in: bf16, %o: bf16):
    // expected-error @below {{contract of pass 5 broken: float value is neither BF16 nor FP8}}
    %e = arith.extf %in : bf16 to f32
    %t = arith.truncf %e : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.return
}
