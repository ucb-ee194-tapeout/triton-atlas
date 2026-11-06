// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-legalize-shapes | FileCheck %s

func.func @single_tile(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%a : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @single_tile
// CHECK: linalg.matmul {{.*}}ttatlas.mxu = 0 : i32

// -----

func.func @wide_tensor(%a: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %cst = arith.constant 0.0 : f32
  // expected-error @+1 {{atlas: unsupported linalg.fill: expected linalg tensors to be 32 or 32x32}}
  %result = linalg.fill ins(%cst : f32) outs(%a : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %result : tensor<64x32xf32>
}

// -----

func.func @conflicting_unit(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @+1 {{atlas: unsupported linalg.matmul: linalg.matmul has conflicting ttatlas.mxu; expected 0}}
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%a : tensor<32x32xf32>) {ttatlas.mxu = 1 : i32} -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
