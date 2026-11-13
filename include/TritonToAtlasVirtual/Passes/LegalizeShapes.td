// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 4 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 4 · ttatlas-legalize-shapes".
def LegalizeShapes : Pass<"ttatlas-legalize-shapes", "mlir::ModuleOp"> {
  let summary = "Tile linalg compute to 32x32 (MLIR's SCF tile-and-fuse + unroll); pad, mask and tag the tiles";
  let description = [{
    Out: every linalg value is one tile (32 or 32x32) at constant extract_slice / insert_slice offsets; a larger
    tt.load is read only through constant tile slices, and a larger stored tensor is built from one insert_slice per
    tile (aligned, distinct, covering) over a tensor.empty. A tiled matmul's K loop, unrolled, is the K chain:
    linalg.fill zero, then linalg.matmuls each taking the previous one's result; every linalg.fill writes a fresh
    tile-sized tensor.empty (upstream's fold-tensor-empty patterns turn slices of a larger empty into small ones).
    A load that is not whole tiles (a 16x16 load) is read by one tensor.pad up to whole tiles, and a stored value that
    is not whole tiles is a constant crop from the origin of a tile or of a tensor built from tiles. Each
    linalg.matmul carries ttatlas.mxu. Each memory boundary tile that holds padded lanes carries its own ttatlas.pad
    (rank 2, that tile's logical shape; none on a full tile) on the op that yields it: the tt.load, the tensor.pad
    or the extract_slice of a load, the producer of a stored tile. A reduction that reads padded lanes not holding its
    identity reads them through a ttatlas.mask_apply linalg.mul or linalg.add by a constant mask tile. Tiling uses
    MLIR's SCF driver (scf::tileConsumerAndFuseProducersUsingSCF, then scf::tileUsingSCF), padding
    linalg::rewriteAsPaddedOp.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect"];
}
