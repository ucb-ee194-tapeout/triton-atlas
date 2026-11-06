// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 4 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 4 · ttatlas-legalize-shapes".
def LegalizeShapes : Pass<"ttatlas-legalize-shapes", "mlir::ModuleOp"> {
  let summary = "Tile linalg compute to 32x32 (MLIR tile-and-fuse + unroll); tag padding and MXU units";
  let description = [{
    Out: every linalg op works on 32x32 tensors at constant extract_slice / insert_slice offsets; a whole-tensor
    tt.load is read only through constant 32x32 slices, and a stored tensor is built only from 32x32 insert_slices.
    A tiled matmul's K loop, unrolled, is the K chain (seeded by linalg.fill zero). Each linalg.matmul carries
    ttatlas.mxu, padded values ttatlas.pad. Uses MLIR's tiling interface instead of a hand-written splitter (IA-19).
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
