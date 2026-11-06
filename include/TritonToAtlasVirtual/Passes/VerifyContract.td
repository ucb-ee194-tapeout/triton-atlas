// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Not one of the pipeline's passes: tests run it after the pass they test, and lanes run it on hand-written
// fixtures, so a fixture claiming to be "pass N's output" really meets pass N's output contract. Contracts:
// triton-atlas design agenda, each pass's page.
def VerifyContract : Pass<"ttatlas-verify-contract", "mlir::ModuleOp"> {
  let summary = "Check that the IR meets the output contract of pass `stage` (1-7)";
  let description = [{
    stage=1: one tt.func; every tt.load / tt.store carries ttatlas.host_dtype.
    stage=2: every tt.dot accumulator is a zero constant, except a later step of a ttatlas.k_chain, which takes
             the previous step's result (same id and len, pos - 1); no scf.for with a constant trip count.
    stage=3: no tt.dot, tt.trans or tt.reduce; no arith / math op on a float tensor; no other tt op except tt.load
             producing a float tensor (constants, loads, stores and integer or pointer address math stay TTIR).
    stage=4: every 2-D linalg value is 32x32; every linalg.matmul carries ttatlas.mxu; a larger tt.load is read
             only through constant 32x32 extract_slices, a larger stored tensor built only from constant 32x32
             insert_slices over a tensor.empty, one per tile, covering it.
    stage=5: linalg.matmul inputs are FP8 E4M3; every other float value or pointer is BF16 or FP8 (a matmul chain's
             accumulator may be f32: the matmuls, their linalg.fill seed, and the zero and tensor.empty only it uses).
    stage=6: every tt.load / tt.store carries #ttatlas.slot.
    stage=7: no TTIR is left; one argument-free top-level func.func carrying ttatlas.kernel, none nested.
  }];
  let options = [Option<"stage", "stage", "unsigned", /*default=*/"0", "pass whose output contract to check (1-7)">];

  // Loads the ttatlas dialect, so the tags it checks are verified too.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
