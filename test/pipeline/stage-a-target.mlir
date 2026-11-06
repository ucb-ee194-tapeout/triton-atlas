// RUN: triton-atlas-opt %S/../fixtures/matmul_relu.ttir.mlir --ttatlas-pipeline | FileCheck %s --implicit-check-not='tt.'
// XFAIL: *

// Stage A's goal: matmul + ReLU becomes Atlas virtual IR. Expected to fail until passes 5 and 6 (and the three small
// first versions in passes 1, 3 and 4) are written; --implicit-check-not makes any TTIR left anywhere fail. Remove the
// XFAIL line when it passes. Op sequence from ttir2linalg ecd55fd examples/ttir_to_atlas/matmul_relu.virtual.mlir; the
// function keeps the kernel's name. The fixture stores f32: ttatlas-validate-ttir records f32 as the store's host
// dtype, and ttatlas-legalize-dtypes retypes the output to BF16.

// CHECK: func.func @matmul_relu_kernel() -> !atlas.virtual_state
// CHECK: atlas.virtual_start
// CHECK: atlas.virtual_input_fp8
// CHECK: atlas.virtual_input_fp8
// CHECK: atlas.virtual_xlu_transpose
// CHECK: atlas.virtual_mxu_matmul
// CHECK: atlas.virtual_vpu_unary
// CHECK-SAME: kind = "relu"
// CHECK: atlas.virtual_output_bf16
// CHECK-SAME: index = 0
