// Pass 4, step 4c: tile the packed block grid. Each packed op is tiled by 1
// on every grid dimension above 1 (a contraction's K included, which chains
// the K steps through the accumulator) and every loop is unrolled, so each op
// is one tile and program order gives a legal schedule.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TILEGRID
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct TileGrid : impl::TileGridBase<TileGrid> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    SmallVector<linalg::LinalgOp> ops;
    module.walk([&](linalg::LinalgOp op) { ops.push_back(op); });
    for (linalg::LinalgOp op : ops) {
      // A packed op has the grid loops first and the 32x32 block loops after.
      SmallVector<int64_t> ranges = op.getStaticLoopRanges();
      SmallVector<OpFoldResult> sizes;
      bool tiled = false;
      for (auto [loop, range] : llvm::enumerate(ranges)) {
        bool grid = loop < ranges.size() / 2 && range > 1;
        tiled |= grid;
        sizes.push_back(rewriter.getIndexAttr(grid ? 1 : 0));
      }
      if (!tiled)
        continue;
      scf::SCFTilingOptions options;
      options.setTileSizes(sizes);
      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> result =
          scf::tileUsingSCF(rewriter, cast<TilingInterface>(*op), options);
      if (failed(result)) {
        ttatlas::emitUnsupported(op, "cannot tile this linalg op by block");
        return signalPassFailure();
      }
      rewriter.replaceOp(op, result->replacements);
    }
    // Innermost loops first, so an outer loop clones loops already unrolled.
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops)
      if (failed(loopUnrollFull(loop))) {
        ttatlas::emitUnsupported(loop, "tile loop with no constant trip count");
        return signalPassFailure();
      }
  }
};

} // namespace
} // namespace mlir::triton
