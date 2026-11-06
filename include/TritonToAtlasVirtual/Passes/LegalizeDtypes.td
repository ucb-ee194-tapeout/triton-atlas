// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 5 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 5 · ttatlas-legalize-dtypes".
def LegalizeDtypes : Pass<"ttatlas-legalize-dtypes", "mlir::ModuleOp"> {
  let summary = "Retype to Atlas dtypes: FP8 matmul operands, BF16 elsewhere";
  let description = [{
    Out: linalg.matmul inputs are FP8 E4M3; all other compute, and the pointers of BF16 outputs, are BF16. A
    matmul chain's accumulator (its linalg.fill seed and the matmuls) is exempt: it stays in the MXU. f32 sources are rejected under the first precision policy.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
