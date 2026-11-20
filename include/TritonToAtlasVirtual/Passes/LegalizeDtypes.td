// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 5 of the TTIR -> Atlas pipeline. Retypes to the dtypes Atlas has.
def LegalizeDtypes : Pass<"ttatlas-legalize-dtypes", "mlir::ModuleOp"> {
  let summary = "Retype to Atlas dtypes: FP8 matmul operands, BF16 elsewhere";
  let description = [{
    Out: linalg.matmul inputs are FP8 E4M3; all other compute, matmul accumulators included, is BF16. tt.load and
    tt.store keep the host's pointer types: a value stored as f32 goes through an explicit arith.extf from BF16. f32
    sources are rejected.
  }];

  // Loads the ttatlas dialect, so the slot attributes later passes attach are verified.
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::linalg::LinalgDialect", "mlir::ttatlas::TTAtlasDialect"];
}
