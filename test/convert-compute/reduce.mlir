// RUN: triton-atlas-opt %s --split-input-file --convert-triton-compute-to-linalg --ttatlas-verify-contract=stage=3 | FileCheck %s

// Reduce group: a row reduction (axis 1) becomes linalg.reduce over dimension 1. tt.reduce has no init value, so the
// output is filled with the combine's identity: -0.0 for an add (the exact one), -inf for a max.

// CHECK-LABEL: tt.func @row_sum
// CHECK: %[[ZERO:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: %[[INIT:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: linalg.reduce { arith.addf } ins(%{{.*}} : tensor<32x32xf32>) outs(%[[INIT]] : tensor<32xf32>) dimensions = [1]
// CHECK-NOT: tt.reduce
tt.func @row_sum(%x: tensor<32x32xf32>) -> tensor<32xf32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<32x32xf32>) -> tensor<32xf32>
  tt.return %r : tensor<32xf32>
}

// -----

// CHECK-LABEL: tt.func @row_max
// CHECK: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[INIT:.*]] = linalg.fill ins(%[[NEG_INF]] : f32)
// CHECK: linalg.reduce { arith.maxnumf } ins(%{{.*}} : tensor<32x32xf32>) outs(%[[INIT]] : tensor<32xf32>) dimensions = [1]
tt.func @row_max(%x: tensor<32x32xf32>) -> tensor<32xf32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %m = arith.maxnumf %a, %b : f32
    tt.reduce.return %m : f32
  }) : (tensor<32x32xf32>) -> tensor<32xf32>
  tt.return %r : tensor<32xf32>
}

// -----

// A min reduction starts from +inf.
// CHECK-LABEL: tt.func @row_min
// CHECK: %[[POS_INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[INIT:.*]] = linalg.fill ins(%[[POS_INF]] : f32)
// CHECK: linalg.reduce { arith.minnumf } ins(%{{.*}} : tensor<32x32xf32>) outs(%[[INIT]] : tensor<32xf32>) dimensions = [1]
tt.func @row_min(%x: tensor<32x32xf32>) -> tensor<32xf32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %m = arith.minnumf %a, %b : f32
    tt.reduce.return %m : f32
  }) : (tensor<32x32xf32>) -> tensor<32xf32>
  tt.return %r : tensor<32xf32>
}
