// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 4 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 4 · ttatlas-legalize-shapes".
def LegalizeShapes : Pass<"ttatlas-legalize-shapes", "mlir::ModuleOp"> {
  let summary = "Tile linalg compute to 32x32 (MLIR tile-and-fuse + unroll); tag padding and MXU units";
  let description = [{
    Out: every linalg value is one tile (32 or 32x32) at constant extract_slice / insert_slice offsets; a larger
    tt.load is read only through constant tile slices, and a larger stored tensor is built from one insert_slice per
    tile (aligned, distinct, covering) over a tensor.empty. A tiled matmul's K loop, unrolled, is the K chain:
    linalg.fill zero, then linalg.matmuls each taking the previous one's result; every linalg.fill writes a fresh
    tile-sized tensor.empty (upstream's fold-tensor-empty patterns turn slices of a larger empty into small ones).
    Each linalg.matmul carries ttatlas.mxu, padded values ttatlas.pad. Uses MLIR's tiling interface instead of a
    hand-written splitter.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
