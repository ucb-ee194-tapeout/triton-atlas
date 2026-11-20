// RUN: triton-atlas-opt %S/../inputs/matmul_relu.ttir.mlir --ttatlas-pipeline | FileCheck %s --implicit-check-not='tt.'

// The full pipeline (the order in lib/TritonToAtlasVirtual/Pipeline.cpp) turns the matmul + ReLU kernel into Atlas
// virtual IR with no TTIR left (--implicit-check-not). The weight arrives as the device's [n][k] tile (its slot says
// transposed), so there is no XLU transpose. The input stores f32; ttatlas-legalize-dtypes retypes the output to BF16.

// CHECK: func.func @matmul_relu_kernel() -> !atlas.virtual_state
// CHECK: atlas.virtual_start
// CHECK: atlas.virtual_input_fp8
// CHECK: atlas.virtual_input_fp8
// CHECK-NOT: atlas.virtual_xlu_transpose
// CHECK: atlas.virtual_mxu_load_weight
// CHECK: atlas.virtual_mxu_reset
// CHECK: atlas.virtual_mxu_readout_bf16
// CHECK: atlas.virtual_vpu_binary
// CHECK-SAME: kind = "max"
// CHECK: atlas.virtual_output_bf16
// CHECK-SAME: index = 0
