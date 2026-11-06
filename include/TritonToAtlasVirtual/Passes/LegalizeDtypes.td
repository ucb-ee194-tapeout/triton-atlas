// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 4 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 4 · ttatlas-legalize-dtypes".
def LegalizeDtypes : Pass<"ttatlas-legalize-dtypes", "mlir::ModuleOp"> {
  let summary = "Retype TTIR to Atlas dtypes: FP8 matmul operands, BF16 elsewhere";
  let description = [{
    Out: matmul operands are FP8 E4M3; all other compute, and the pointers of BF16 outputs, are BF16. Accumulators
    inside a ttatlas.k_chain are exempt. f32 sources are rejected under the first precision policy.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
