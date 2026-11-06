// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Not one of the pipeline's passes: tests run it after the pass they test, and lanes run it on hand-written
// fixtures, so a fixture claiming to be "pass N's output" really meets pass N's output contract. Contracts:
// triton-atlas design agenda, each pass's page.
def VerifyContract : Pass<"ttatlas-verify-contract", "mlir::ModuleOp"> {
  let summary = "Check that the IR meets the output contract of pass `stage` (1-6)";
  let description = [{
    stage=1: one tt.func; every tt.load / tt.store carries ttatlas.host_dtype.
    stage=2: every tt.dot accumulator is a zero constant, except a later step of a ttatlas.k_chain, which takes
             the previous step's result (same id and len, pos - 1); no scf.for with a constant trip count.
    stage=3: every 2-D value (results and block arguments) is 32x32; every tt.dot carries ttatlas.mxu.
    stage=4: tt.dot operands are FP8 E4M3; every other float value or pointer is BF16 or FP8 (K-chain dots and
             the zero constant that starts a chain may be f32).
    stage=5: every tt.load / tt.store carries #ttatlas.slot.
    stage=6: no TTIR is left; one argument-free top-level func.func carrying ttatlas.kernel, none nested.
  }];
  let options = [Option<"stage", "stage", "unsigned", /*default=*/"0", "pass whose output contract to check (1-6)">];

  // Loads the ttatlas dialect, so the tags it checks are verified too.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
