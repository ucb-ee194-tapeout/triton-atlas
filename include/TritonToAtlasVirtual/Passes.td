// Passes that lower Triton IR (TTIR) to Atlas virtual ops (atlas-mlir's `atlas.virtual_*`).
// Layout follows Triton's AMD backend (triton v3.9.0 third_party/amd/include/TritonAMDGPUToLLVM/Passes.td).
#ifndef TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES
#define TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToAtlasVirtual : Pass<"convert-triton-to-atlas-virtual", "mlir::ModuleOp"> {
  let summary = "Convert TTIR kernels to Atlas virtual ops";
  let description = [{
    Rewrites each `tt.func` into a `func.func` of Atlas virtual ops on 32x32 tiles: tile loads and stores become
    `atlas.virtual_input_*` / `atlas.virtual_output_bf16`, `tt.dot` becomes an MXU matmul, and elementwise ops
    become VPU ops. Anything it can't map is an error naming the op.
  }];

  // The ops this pass creates; MLIR loads these dialects before the pass runs.
  let dependentDialects = ["mlir::atlas::AtlasDialect",
                           "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect"];
}

#endif // TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES
