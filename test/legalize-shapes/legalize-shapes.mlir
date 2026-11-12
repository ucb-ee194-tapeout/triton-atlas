// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-legalize-shapes --canonicalize | FileCheck %s

func.func @single_tile(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @single_tile
// CHECK: linalg.matmul {{.*}}ttatlas.mxu = 0 : i32

// -----

func.func @non_tile_multiple(%a: tensor<50x32xf32>) -> tensor<50x32xf32> {
  %cst = arith.constant 0.0 : f32
  // expected-error @+1 {{atlas: unsupported linalg.fill: iteration dimensions must be static multiples of 32}}
  %result = linalg.fill ins(%cst : f32) outs(%a : tensor<50x32xf32>) -> tensor<50x32xf32>
  return %result : tensor<50x32xf32>
}

// -----

func.func @conflicting_unit(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @+1 {{atlas: unsupported linalg.matmul: linalg.matmul has conflicting ttatlas.mxu; expected 0}}
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%a : tensor<32x32xf32>) {ttatlas.mxu = 1 : i32} -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

func.func @tile_add(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %empty = tensor.empty() : tensor<64x64xbf16>
  %result = linalg.add ins(%a, %b : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%empty : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %result : tensor<64x64xbf16>
}

// CHECK-LABEL: func.func @tile_add
// CHECK: tensor.extract_slice %a[0, 0] [32, 32] [1, 1]
// CHECK-COUNT-4: linalg.add
// CHECK: tensor.insert_slice

// -----

func.func @split_k_matmul(%a: tensor<32x64xf8E4M3FN>, %b: tensor<64x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x64xf8E4M3FN>, tensor<64x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @split_k_matmul
// CHECK: linalg.fill
// CHECK-COUNT-2: linalg.matmul {{.*}}ttatlas.mxu = 0 : i32

// -----

func.func @reduce_across_tiles(%input: tensor<32x64xf32>, %init: tensor<32xf32>) -> tensor<32xf32> {
  %result = linalg.reduce ins(%input : tensor<32x64xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%lhs: f32, %rhs: f32) {
    %sum = arith.addf %lhs, %rhs : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<32xf32>
}

// CHECK-LABEL: func.func @reduce_across_tiles
// CHECK-COUNT-2: linalg.reduce

// -----

tt.func @masked_tile(%ptr: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limit = arith.constant 16 : i32
  %limits = tt.splat %limit : i32 -> tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %row_mask = tt.expand_dims %cmp {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
  %mask = tt.broadcast %row_mask : tensor<32x1xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %a = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %b = tt.load %ptrs : tensor<32x32x!tt.ptr<f32>>
  %empty = tensor.empty() : tensor<32x32xf32>
  %scalar_zero = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%scalar_zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.store %ptrs, %result, %mask : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func @masked_tile
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>
// CHECK: linalg.matmul {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>
