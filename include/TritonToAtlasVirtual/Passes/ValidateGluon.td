// Gluon frontend gate.  This intentionally runs before the shared TTIR passes:
// Gluon may carry layout anchors which are not part of TTIR's Atlas contract.
def ValidateGluon : Pass<"ttatlas-validate-gluon", "mlir::ModuleOp"> {
  let summary = "Accept only Gluon IR that the Atlas layout-normalization pass can erase safely";
  let description = [{
    Out: unchanged, validated Gluon/TTIR.  The pass admits ordinary TTIR plus
    gluon.set_auto_layout anchors over static rank-1 or rank-2 tensors.  It
    rejects GPU-dialect operations and any other Gluon operation: Atlas has no
    semantics for them before layout normalization.  It does not select Atlas
    layouts, change types, assign slots, or lower compute.
  }];
  let dependentDialects = [
    "mlir::triton::gluon::GluonDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}
