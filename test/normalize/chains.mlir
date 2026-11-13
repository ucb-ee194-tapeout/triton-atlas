// RUN: triton-atlas-opt %s --split-input-file --ttatlas-normalize --ttatlas-verify-contract=stage=2 | FileCheck %s

// How pass 2 finds K chains, tags them and moves a bias: a bias that seeds a K loop, chains with no loop left, several
// chains, a partial sum used twice, nested loops and a lone dot. The stage-2 contract check runs after it on each case.

// A bias that seeds a K loop (acc = broadcast(bias) before the loop).
// CHECK-LABEL: tt.func @seeded_loop
// CHECK: %[[Z:.*]] = arith.constant dense<0.000000e+00>
// CHECK: %[[D0:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[Z]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 4>}
// CHECK: %[[D1:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D0]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>}
// CHECK: %[[D2:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D1]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 2, len = 4>}
// CHECK: %[[D3:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D2]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 3, len = 4>}
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[D3]], %arg2
// CHECK: tt.return %[[SUM]]
tt.func @seeded_loop(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  %acc = scf.for %k = %c0 to %c4 step %c1 iter_args(%s = %bias) -> (tensor<32x32xf32>) : i32 {
    %d = tt.dot %a, %b, %s : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    scf.yield %d : tensor<32x32xf32>
  }
  tt.return %acc : tensor<32x32xf32>
}

// -----

// A loop Triton already unrolled by 2 (tl.range(..., loop_unroll_factor=2)): two chained dots per iteration.
// CHECK-LABEL: tt.func @unrolled_by_two
// CHECK-NOT: scf.for
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 4>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 2, len = 4>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 3, len = 4>}
// CHECK-NOT: arith.addf
// CHECK: tt.return
tt.func @unrolled_by_two(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %c1 = arith.constant 1 : i32
  %acc = scf.for %k = %c0 to %c2 step %c1 iter_args(%s = %zero) -> (tensor<32x32xf32>) : i32 {
    %d0 = tt.dot %a, %b, %s : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    %d1 = tt.dot %b, %a, %d0 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    scf.yield %d1 : tensor<32x32xf32>
  }
  tt.return %acc : tensor<32x32xf32>
}

// -----

// dot(x, w) + dot(w, x), which Triton's combine folds into one dot accumulating onto the other.
// CHECK-LABEL: tt.func @dot_plus_dot
// CHECK: %[[D0:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %[[D0]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>}
tt.func @dot_plus_dot(%x: tensor<32x32xf8E4M3FN>, %w: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %x, %w, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %d1 = tt.dot %w, %x, %d0 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %d1 : tensor<32x32xf32>
}

// -----

// Two independent chains in one function get ids 0 and 1.
// CHECK-LABEL: tt.func @two_chains
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 1, pos = 0, len = 2>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 1, pos = 1, len = 2>}
tt.func @two_chains(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %p0 = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %p1 = tt.dot %a, %b, %p0 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %q0 = tt.dot %b, %a, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %q1 = tt.dot %b, %a, %q0 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %p1, %q1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----

// A partial sum used twice (d1 is also returned) ends its chain: [d0, d1] and [d2], with d1 added after d2.
// CHECK-LABEL: tt.func @partial_sum_used_twice
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>}
// CHECK: %[[D1:.*]] = tt.dot %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>}
// CHECK: %[[Z:.*]] = arith.constant dense<0.000000e+00>
// CHECK: %[[D2:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[Z]] :
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[D2]], %[[D1]]
// CHECK: tt.return %[[D1]], %[[SUM]]
tt.func @partial_sum_used_twice(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %d1 = tt.dot %a, %b, %d0 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %d2 = tt.dot %a, %b, %d1 : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %d1, %d2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----

// Nested constant-trip loops carrying one accumulator unroll into one chain of 2 x 2 = 4 dots.
// CHECK-LABEL: tt.func @nested
// CHECK-NOT: scf.for
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 4>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 2, len = 4>}
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 3, len = 4>}
// CHECK-NOT: arith.addf
tt.func @nested(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %c1 = arith.constant 1 : i32
  %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%s = %zero) -> (tensor<32x32xf32>) : i32 {
    %in = scf.for %j = %c0 to %c2 step %c1 iter_args(%t = %s) -> (tensor<32x32xf32>) : i32 {
      %d = tt.dot %a, %b, %t : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
      scf.yield %d : tensor<32x32xf32>
    }
    scf.yield %in : tensor<32x32xf32>
  }
  tt.return %acc : tensor<32x32xf32>
}

// -----

// A lone dot from zero: no tag, no add.
// CHECK-LABEL: tt.func @zero_start
// CHECK: tt.dot
// CHECK-NOT: ttatlas.k_chain
// CHECK-NOT: arith.addf
// CHECK: tt.return
tt.func @zero_start(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return %d : tensor<32x32xf32>
}

// -----

// A constant loop that runs zero times is replaced by its initial value; no scf.for is left.
// CHECK-LABEL: tt.func @zero_trip
// CHECK-NOT: scf.for
// CHECK-NOT: tt.dot
// CHECK: tt.return %arg2
tt.func @zero_trip(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %init: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  %acc = scf.for %k = %c4 to %c4 step %c1 iter_args(%s = %init) -> (tensor<32x32xf32>) : i32 {
    %d = tt.dot %a, %b, %s : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    scf.yield %d : tensor<32x32xf32>
  }
  tt.return %acc : tensor<32x32xf32>
}
