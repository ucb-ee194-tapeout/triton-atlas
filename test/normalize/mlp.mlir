// RUN: triton-atlas-opt %S/../fixtures/mlp.ttir.mlir --ttatlas-normalize --ttatlas-verify-contract=stage=2 | FileCheck %s

// Triton's combine folded the MLP's bias into the dot's accumulator. Pass 2 undoes it: the dot starts from zero and
// the bias is added after it, before the ReLU. The stage-2 contract check runs after it.

// CHECK: tt.func public @mlp
// CHECK: %[[BIAS:.*]] = tt.broadcast %{{.*}} : tensor<1x32xf32> -> tensor<32x32xf32>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
// CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[ZERO]]
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[DOT]], %[[BIAS]] : tensor<32x32xf32>
// CHECK: arith.maxnumf %[[SUM]], %{{.*}} : tensor<32x32xf32>
// CHECK: tt.store
