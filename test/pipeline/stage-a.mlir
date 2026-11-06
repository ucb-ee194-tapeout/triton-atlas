// RUN: triton-atlas-opt %S/../fixtures/matmul_relu.ttir.mlir --ttatlas-pipeline | FileCheck %s

// The full pipeline (the order in lib/TritonToAtlasVirtual/Pipeline.cpp) runs on stage A's kernel without error.
// It only checks the kernel survives, TTIR or Atlas, so it keeps passing while the lanes land;
// stage-a-target.mlir checks what the output should become.

// CHECK: {{tt.func public|func.func}} @matmul_relu_kernel
