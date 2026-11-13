// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --verify-diagnostics

// Accumulators pass 5 can't keep in the MXU: both would need f32 outside it.

// A matmul starting from a bias instead of linalg.fill (what pass 2's experimental fold-bias-into-dot option leaves).
tt.func @bias_accumulator(%a: tensor<32x32xf8E4M3FN>, %bias: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported linalg.matmul: the accumulator neither starts from linalg.fill nor continues a matmul chain}}
  %m = linalg.matmul ins(%a, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%bias : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}

// -----

// A partial sum of a chain that is also used outside it.
tt.func @partial_sum(%a: tensor<32x32xf8E4M3FN>, %p: tensor<32x32x!tt.ptr<f32>>) {
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  // expected-error @below {{atlas: unsupported linalg.matmul: a partial sum of the matmul chain is used outside it; it stays in the MXU}}
  %m0 = linalg.matmul ins(%a, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m1 = linalg.matmul ins(%a, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.store %p, %m0 : tensor<32x32x!tt.ptr<f32>>
  tt.return
}
