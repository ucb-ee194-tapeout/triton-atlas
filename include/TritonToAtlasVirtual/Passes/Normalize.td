// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 2 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 2 · ttatlas-normalize".
def Normalize : Pass<"ttatlas-normalize", "mlir::ModuleOp"> {
  let summary = "Put TTIR into the forms later passes expect";
  let description = [{
    Out: every tt.dot accumulator is zero or the previous dot of the same K chain; a bias moves to a separate add
    (undoing Triton's combine); unrolled K-loop dots carry ttatlas.k_chain; no scf.for with a constant trip count.
  }];

  // Experimental, off by default: atlas-mlir lists VMATPUSH.ABF16 (virtual_mxu_load_acc_bf16) as Untested
  // (docs/hardware-status.md:52 @ 8ee175c), and nothing lowers the folded form yet: the stage-2 contract
  // (ttatlas-verify-contract) expects every chain to start from zero, pass 7 has no load-accumulator pattern, and
  // load_acc takes a BF16 tile while the bias here is f32. Pass 3 already handles it (the matmul's output is the bias).
  // Two switches, either of which turns the fold on: this pass option (for lit tests and triton-atlas-opt) and the
  // module tag ttatlas.fold_bias_into_dot, which compiler.py sets from AtlasOptions.fold_bias_into_dot.
  let options = [Option<"foldBiasIntoDot", "fold-bias-into-dot", "bool", /*default=*/"false",
                        "experimental, not lowered yet: keep a dot's bias as its accumulator, to seed the MXU with "
                        "VMATPUSH.ABF16, instead of moving it to an add after the chain">];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified, and arith, whose
  // constants and adds the bias unfold creates.
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::ttatlas::TTAtlasDialect"];
}
