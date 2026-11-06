// Passes that lower Triton IR (TTIR) to Atlas virtual ops (atlas-mlir's `atlas.virtual_*`).
// Layout follows Triton's AMD backend (triton v3.9.0 third_party/amd/include/TritonAMDGPUToLLVM/Passes.td).
#ifndef TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES
#define TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToAtlasVirtual : Pass<"convert-triton-to-atlas-virtual", "mlir::ModuleOp"> {
  let summary = "Convert TTIR kernels to Atlas virtual ops";
  let description = [{
    Lowers the initial static two-layer 32x32 FP8 MLP case from one `tt.func` to a `func.func` Atlas virtual stream.
    Unsupported functions and operation patterns fail with diagnostics; general elementwise, attention, and
    multi-kernel module lowering are not implemented yet.
  }];

  // The ops this pass creates; MLIR loads these dialects before the pass runs.
  let dependentDialects = ["mlir::atlas::AtlasDialect",
                           "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect"];
}

def LegalizeAtlasShapes : Pass<"ttatlas-legalize-shapes", "mlir::ModuleOp"> {
  let summary = "Validate Atlas compute tile shapes and select MXU0";
  let description = [{
    First shape-legalization slice: require every tensor value used by linalg
    compute operations to be 32x32, and tag each linalg.matmul with
    ttatlas.mxu = 0. Tiling, padding, reductions, and slice reassembly are
    added in later slices.
  }];
}

#endif // TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES
