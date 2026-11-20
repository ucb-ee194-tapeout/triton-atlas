// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 4 of the TTIR -> Atlas pipeline, in small steps (4a, 4b, 4c, 4d). ttatlas-legalize-shapes is the pipeline
// that runs them with canonicalize and CSE between (Pipeline.cpp); 4d is upstream's linalg-fold-unit-extent-dims.
def PackBlocks : Pass<"ttatlas-pack-blocks", "mlir::ModuleOp"> {
  let summary = "Step 4a: pack every linalg op into 32x32 tiles; clamp the lanes a padded reduction would read";
  let description = [{
    Generalizes the named linalg ops, folds a load's transpose into the ops that read it, then packs every
    linalg.generic into 32-blocks with upstream linalg::pack (matmul B comes out packed transposed; no
    outer_dims_perm). A pack of a tensor.empty becomes a tensor.empty and a pack of a splat constant a splat
    constant. Before every BF16 row reduction over a padded column dimension a generic clamps the operand with
    arith.minnumf / arith.maxnumf against raw-bit constant tiles: lanes inside the Triton block (masked lanes
    included) stay bit-identical (0x7FFF / 0xFFFF), lanes outside it become the reduction's identity. A
    contraction's K padding is the pack's zero. Rejects a padded K whose operand is not a load, and a padded
    reduction it has no rule for.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect", "mlir::arith::ArithDialect"];
}

def FoldPackPairs : Pass<"ttatlas-fold-pack-pairs", "mlir::ModuleOp"> {
  let summary = "Step 4b: fold every linalg.unpack -> linalg.pack pair with the same layout";
  let description = [{
    A pack of an unpack with the same inner_dims_pos, inner_tiles and types is the unpack's source, padded or not:
    pad lanes are don't-care except where step 4a clamped them. One pack per load and one unpack per store remain.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def TileGrid : Pass<"ttatlas-tile-grid", "mlir::ModuleOp"> {
  let summary = "Step 4c: tile the packed block grid by 1 and unroll every loop";
  let description = [{
    scf::tileUsingSCF on each packed linalg op with tile size 1 on every grid dimension above 1 (a contraction's K
    included, which chains the K steps through the accumulator) and 0 elsewhere, then a full unroll of every
    scf.for. Each op ends as one 32x32 tile (3-D for a contraction); a row that spans several tiles is a chain of
    reductions passing the partial through the init.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect"];
}
