// RUN: triton-atlas-opt %S/../inputs/matmul_relu.ttir.mlir --ttatlas-pipeline | FileCheck %s

// The full pipeline (the order in lib/TritonToAtlasVirtual/Pipeline.cpp) runs on stage A's kernel without error.
// Pass 7 now lowers the whole kernel, so it ends as an Atlas function with a matmul and a ReLU, and no TTIR left;
// stage-a-target.mlir checks what the output should become.

// CHECK: func.func public @matmul_relu_kernel
// CHECK: atlas.virtual_mxu_matmul
// CHECK: atlas.virtual_vpu_unary
// CHECK-SAME: kind = "relu"
// CHECK: atlas.virtual_output_bf16
