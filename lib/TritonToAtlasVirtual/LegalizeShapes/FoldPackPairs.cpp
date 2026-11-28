// Pass 4, step 4b: fold every unpack -> pack pair between compute ops.
// Compute stays packed from load to store, so the boundary is one pack per
// load and one unpack per store. A pair with padded lanes is not an identity
// in general, but pad lanes are don't-care: step 4a already clamped the ones a
// padded reader could carry into a valid lane.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iterator>

namespace mlir::triton {

#define GEN_PASS_DEF_FOLDPACKPAIRS
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

// pack(unpack(x)) with the same layout is x. Unlike upstream's pattern, this
// one also folds a pack that pads.
struct FoldUnpackPack : OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PackOp pack,
                                PatternRewriter &rewriter) const override {
    auto unpack = pack.getSource().getDefiningOp<linalg::UnPackOp>();
    if (!unpack || unpack.getSource().getType() != pack.getDestType() ||
        unpack.getInnerDimsPos() != pack.getInnerDimsPos() ||
        unpack.getOuterDimsPerm() != pack.getOuterDimsPerm() ||
        unpack.getMixedTiles() != pack.getMixedTiles())
      return failure();
    rewriter.replaceOp(pack, unpack.getSource());
    return success();
  }
};

struct FoldPackPairs : impl::FoldPackPairsBase<FoldPackPairs> {
  void runOnOperation() override {
    // The MLIR linked into Triton's Python wheel has a different linalg
    // region trait TypeID from the plugin's linalg dialect.  Running the
    // greedy pack(unpack) fold there can erase the body of every generic as
    // it replaces the pair.  Keep the pair in the plugin runtime; emission
    // already handles the boundary ops and the standalone tool still gets
    // the intended fold.
    if (std::getenv("TTATLAS_PLUGIN_RUNTIME"))
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldUnpackPack>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
    if (std::getenv("TTATLAS_TRACE_BODY"))
      getOperation()->walk([&](linalg::GenericOp op) {
        llvm::errs() << "ttatlas FoldPackPairs generic body="
                     << std::distance(op.getBody()->begin(), op.getBody()->end())
                     << "\n";
      });
  }
};

} // namespace
} // namespace mlir::triton
