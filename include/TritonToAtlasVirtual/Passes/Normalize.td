// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 2 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 2 · ttatlas-normalize".
def Normalize : Pass<"ttatlas-normalize", "mlir::ModuleOp"> {
  let summary = "Put TTIR into the forms later passes expect";
  let description = [{
    Out: every tt.dot accumulator is zero or the previous dot of the same K chain; a bias moves to a separate add
    (undoing Triton's combine); unrolled K-loop dots carry ttatlas.k_chain; no scf.for with a constant trip count.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified, and arith, whose
  // constants and adds the bias unfold creates.
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::ttatlas::TTAtlasDialect"];
}
