// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Not one of the pipeline's passes: tests run it after the pass they test, and lanes run it on hand-written
// fixtures, so a fixture claiming to be "pass N's output" really meets pass N's output contract. Each pass's contract is
// the checks listed per stage below.
def VerifyContract : Pass<"ttatlas-verify-contract", "mlir::ModuleOp"> {
  let summary = "Check that the IR meets the output contract of pass `stage` (1-7)";
  let description = [{
    stage=1: one tt.func.
    stage=2: every tt.dot accumulator is a zero constant, except a later step of a ttatlas.k_chain, which takes
             the previous step's result (same id and len, pos - 1); no scf.for with a constant trip count.
    stage=3: no tt.dot, tt.trans or tt.reduce; no arith / math op on a float tensor; no other tt op except tt.load
             producing a float tensor (constants, loads, stores and integer or pointer address math stay TTIR).
    stage=4: every linalg value is one tile (32 or 32x32); every linalg.pack reads a tt.load (through float casts) and
             every linalg.unpack is read only by a tt.store (through float casts); no scf.for;
             a tt.load is read only by a pack (or a copy's tt.store).
    stage=5: linalg.matmul inputs are FP8 E4M3; every other float value is BF16 or FP8, accumulators included, except
             a cast read only as a store's value (a store keeps the host's pointer type).
    stage=6: every pack, unpack, copy load/store and non-splat constant read by a generic carries #ttatlas.slot; no
             tt.load / tt.store is nested.
    stage=7: no TTIR is left; one argument-free top-level func.func carrying ttatlas.kernel, none nested.
  }];
  let options = [Option<"stage", "stage", "unsigned", /*default=*/"0", "pass whose output contract to check (1-7)">];

  // Loads the ttatlas dialect, so the tags it checks are verified too.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
