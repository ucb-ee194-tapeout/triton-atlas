// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 5 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 5 · ttatlas-legalize-dtypes".
def LegalizeDtypes : Pass<"ttatlas-legalize-dtypes", "mlir::ModuleOp"> {
  let summary = "Retype to Atlas dtypes: FP8 matmul operands, BF16 elsewhere";
  let description = [{
    Out: linalg.matmul inputs are FP8 E4M3; all other compute, and the pointers of BF16 outputs, are BF16. A
    matmul chain's accumulator (its linalg.fill seed and the matmuls) is exempt: it stays in the MXU, and one
    linalg.generic{arith.truncf} after each chain's last matmul marks its readout to BF16. f32 sources are rejected
    under the first precision policy.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified, and the dialects
  // of the readout marker it creates.
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect",
                           "mlir::ttatlas::TTAtlasDialect"];
}
