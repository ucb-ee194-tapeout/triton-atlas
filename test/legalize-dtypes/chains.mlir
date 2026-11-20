// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --ttatlas-verify-contract=stage=5 | FileCheck %s

// Matmul chains and arguments through pass 5, each followed by the stage-5 contract check.

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// A 2-matmul K chain with an f32 output: both matmuls and the fill are BF16, with no readout, and the result is
// stored through an extf to the host's f32.
// CHECK-LABEL: tt.func @k_chain_f32_out
// CHECK: %[[M0:.*]] = linalg.matmul {{.*}} -> tensor<32x32xbf16>
// CHECK-NEXT: %[[M1:.*]] = linalg.matmul {{.*}} outs(%[[M0]] : tensor<32x32xbf16>) -> tensor<32x32xbf16>
// CHECK-NOT: linalg.
// CHECK: %[[W:.*]] = arith.extf %[[M1]] : tensor<32x32xbf16> to tensor<32x32xf32>
// CHECK: tt.store %{{.*}}, %[[W]] : tensor<32x32x!tt.ptr<f32>>
tt.func @k_chain_f32_out(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %p: tensor<32x32x!tt.ptr<f32>>) {
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m0 = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m1 = linalg.matmul ins(%b, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.store %p, %m1 : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

// An unused scalar f32 argument (Inductor passes such sizes and scales) keeps its type: it is the host's value.
// CHECK-LABEL: tt.func @scalar_arg(%arg0: f32,
tt.func @scalar_arg(%s: f32, %x: tensor<32x32xbf16>, %p: tensor<32x32x!tt.ptr<bf16>>) {
  tt.store %p, %x : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
