// RUN: triton-atlas-opt %s --split-input-file --convert-triton-compute-to-linalg --ttatlas-verify-contract=stage=3 | FileCheck %s

// Matmul group: tt.dot becomes linalg.fill zero + linalg.matmul, tt.trans becomes linalg.transpose. Input is pass 2's
// form (hand-written): accumulators zero, or the previous dot of a tagged K chain.

// CHECK-LABEL: tt.func @dot_trans
// The zero tensor the dot started from is dead after the fill replaces it, and is erased.
// CHECK-NOT: arith.constant dense
// CHECK-NOT: tt.trans
// CHECK: %[[WT:.*]] = linalg.transpose ins(%{{.*}} : tensor<32x32xf8E4M3FN>) outs(%{{.*}} : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x32xf32>)
// CHECK-NOT: tt.dot
// CHECK: linalg.matmul ins(%{{.*}}, %[[WT]] : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%[[FILL]] : tensor<32x32xf32>)
// CHECK-NOT: tt.dot
// CHECK: tt.return
tt.func @dot_trans(%x: tensor<32x32xf8E4M3FN>, %w: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
  %d = tt.dot %x, %wt, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %d : tensor<32x32xf32>
}

// -----

// A tagged K chain becomes one fill -> matmul -> matmul def-use chain; the k_chain tag ends here.
// CHECK-LABEL: tt.func @k_chain
// CHECK: %[[FILL:.*]] = linalg.fill
// CHECK: %[[M0:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : {{.*}}) outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT: %[[M1:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : {{.*}}) outs(%[[M0]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT: tt.return %[[M1]]
tt.func @k_chain(%a0: tensor<32x32xf8E4M3FN>, %b0: tensor<32x32xf8E4M3FN>, %a1: tensor<32x32xf8E4M3FN>, %b1: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a0, %b0, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %d1 = tt.dot %a1, %b1, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %d1 : tensor<32x32xf32>
}

// -----

// A dot whose accumulator is a bias (pass 2 keeps it there with its experimental fold option): the bias becomes the
// matmul's output operand, so the matmul adds onto it; no fill.
// CHECK-LABEL: tt.func @bias_accumulator
// CHECK-NOT: linalg.fill
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%arg2 : tensor<32x32xf32>)
// CHECK-NOT: tt.dot
// CHECK: tt.return
tt.func @bias_accumulator(%x: tensor<32x32xf8E4M3FN>, %w: tensor<32x32xf8E4M3FN>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %d = tt.dot %x, %w, %bias : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %d : tensor<32x32xf32>
}
