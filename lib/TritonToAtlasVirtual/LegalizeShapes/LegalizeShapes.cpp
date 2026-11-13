// Pass 4: tile linalg at the Atlas tile size (MLIR's SCF tile-and-fuse driver),
// fully unroll the static tile loops, and attach the MXU policy tag.
#include "LegalizeShapes.h"

#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZESHAPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

static bool isInsideLinalgBody(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getDialectNamespace() == "linalg")
      return true;
  return false;
}

static bool isLinalgCompute(Operation *op) {
  return op->getName().getDialectNamespace() == "linalg" &&
         !isInsideLinalgBody(op);
}

static bool isSupportedStaticDomain(linalg::LinalgOp op) {
  if (op.hasDynamicShape())
    return false;
  for (int64_t size : op.getStaticLoopRanges())
    if (size <= 0)
      return false;
  return true;
}

static bool isSupportedTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || (tensor.getRank() != 1 && tensor.getRank() != 2))
    return false;
  return llvm::all_of(tensor.getShape(), [](int64_t size) {
    return !ShapedType::isDynamic(size) && size > 0;
  });
}

// Forget ops the rewriter erased, so a worklist never touches a dead pointer.
struct LiveOps : RewriterBase::Listener {
  DenseSet<Operation *> live;
  void notifyOperationErased(Operation *op) override { live.erase(op); }
};

struct LegalizeShapes : impl::LegalizeShapesBase<LegalizeShapes> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    DenseMap<Operation *, Attribute> storePadTags;
    if (failed(tagMaskedBoundaries(module, storePadTags))) {
      signalPassFailure();
      return;
    }
    bool hadFailure = false;
    SmallVector<Operation *> computeOps;
    module.walk([&](Operation *op) {
      if (isLinalgCompute(op))
        computeOps.push_back(op);
    });

    // Validate the whole input before rewriting so an unsupported later op
    // cannot leave a partly legalized module behind.
    for (Operation *op : computeOps) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp) {
        ttatlas::emitUnsupported(op, "operation does not implement LinalgOp");
        hadFailure = true;
        continue;
      }
      if (!isSupportedStaticDomain(linalgOp)) {
        ttatlas::emitUnsupported(
            op, "iteration dimensions must be positive and static");
        hadFailure = true;
        continue;
      }
      bool validTensorShapes = true;
      for (Type type : llvm::concat<const Type>(op->getOperandTypes(),
                                                op->getResultTypes())) {
        if (isa<TensorType>(type) && !isSupportedTensor(type)) {
          ttatlas::emitUnsupported(op, "linalg tensor must have static rank 1 "
                                       "or 2 and positive dimensions");
          hadFailure = true;
          validTensorShapes = false;
          break;
        }
      }
      if (!validTensorShapes)
        continue;
      if (op->getName().getStringRef() == "linalg.matmul" &&
          hasConflictingMxu(op))
        hadFailure = true;
    }
    if (hadFailure) {
      signalPassFailure();
      return;
    }

    LiveOps tracker;
    IRRewriter rewriter(module.getContext(), &tracker);
    for (Operation *op : computeOps)
      tracker.live.insert(op);

    // Split every linalg op larger than a tile, then re-queue the tiles: the
    // worklist ends when each op is one tile. A tile can still be too large in
    // a dimension that tiling left whole (a reduction, or an op fused into a
    // tiled consumer), so this repeats per op until it is not.
    SmallVector<Operation *> worklist;
    auto drain = [&]() -> LogicalResult {
      while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        if (!tracker.live.contains(op) || !exceedsTile(op))
          continue;
        auto linalgOp = cast<linalg::LinalgOp>(op);
        scf::SCFTilingOptions options;
        options.setTileSizes(
            atlasTileSizes(rewriter, linalgOp, /*parallelOnly=*/false));
        rewriter.setInsertionPoint(op);
        FailureOr<scf::SCFTilingResult> tiled =
            scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), options);
        if (failed(tiled)) {
          ttatlas::emitUnsupported(op,
                                   "MLIR failed to tile the linalg operation");
          return failure();
        }
        ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                    "tile linalg compute to 32x32");
        rewriter.replaceOp(op, tiled->replacements);
        for (Operation *tile : tiled->tiledOps) {
          tracker.live.insert(tile);
          worklist.push_back(tile);
        }
      }
      return success();
    };

    // Consumers first, so each producer is fused into its consumer's tiles
    // before it is looked at on its own.
    for (Operation *op : llvm::reverse(computeOps)) {
      if (!tracker.live.contains(op))
        continue;
      if (llvm::all_of(op->getResults(),
                       [](Value result) { return result.use_empty(); })) {
        rewriter.eraseOp(op);
        continue;
      }
      auto linalgOp = cast<linalg::LinalgOp>(op);
      if (op->getName().getStringRef() == "linalg.matmul" && !setMxuZero(op)) {
        signalPassFailure();
        return;
      }
      bool needsPadding =
          llvm::any_of(linalgOp.getStaticLoopRanges(), [](int64_t size) {
            return size % ttatlas::target::kTileSize != 0;
          });
      if (needsPadding) {
        linalg::LinalgOp paddedOp;
        SmallVector<Value> replacements;
        rewriter.setInsertionPoint(op);
        if (failed(
                padToAtlasTiles(rewriter, linalgOp, paddedOp, replacements))) {
          signalPassFailure();
          return;
        }
        rewriter.replaceOp(op, replacements);
        op = paddedOp.getOperation();
        tracker.live.insert(op);
        linalgOp = paddedOp;
      }
      if (!exceedsTile(op))
        continue;

      SmallVector<OpFoldResult> sizes =
          atlasTileSizes(rewriter, linalgOp, /*parallelOnly=*/true);
      if (llvm::all_of(sizes,
                       [](OpFoldResult size) { return isZeroInteger(size); })) {
        // Only reductions are too large: nothing to fuse into, split them
        // below.
        worklist.push_back(op);
      } else {
        scf::SCFTileAndFuseOptions options;
        scf::SCFTilingOptions tiling;
        tiling.setTileSizes(sizes);
        options.setTilingOptions(tiling);
        options.setFusionControlFn(fuseLinalgProducersOnly);
        rewriter.setInsertionPoint(op);
        FailureOr<scf::SCFTileAndFuseResult> tiled =
            scf::tileConsumerAndFuseProducersUsingSCF(
                rewriter, cast<TilingInterface>(op), options);
        if (failed(tiled)) {
          ttatlas::emitUnsupported(op,
                                   "MLIR failed to tile the linalg operation");
          signalPassFailure();
          return;
        }
        ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                    "tile linalg compute to 32x32");
        for (auto [original, replacement] : tiled->replacements)
          rewriter.replaceAllUsesWith(original, replacement);
        // The consumer's results were replaced; drop it if nothing else needs
        // it.
        if (llvm::all_of(op->getResults(),
                         [](Value r) { return r.use_empty(); }))
          rewriter.eraseOp(op);
        for (Operation *tile : tiled->tiledAndFusedOps) {
          tracker.live.insert(tile);
          worklist.push_back(tile);
        }
      }
      if (failed(drain())) {
        signalPassFailure();
        return;
      }
    }

    if (failed(unrollTileLoops(module))) {
      signalPassFailure();
      return;
    }
    freshFillDestinations(module, rewriter);

    // Tiling the consumer first and then the producer again slices a loaded
    // tensor twice (64x64 -> 32x64 -> 32x32). Merge each chain into one
    // constant slice of the load, which the stage-4 contract asks for, with
    // MLIR's own patterns; no folding, so nothing else changes here.
    RewritePatternSet mergeSlices(module.getContext());
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(mergeSlices);
    if (failed(applyPatternsGreedily(
            module, std::move(mergeSlices),
            GreedyRewriteConfig().enableFolding(false).enableConstantCSE(
                false)))) {
      module.emitError("atlas: merging tile slices did not converge");
      signalPassFailure();
      return;
    }

    if (failed(propagateStorePadTags(storePadTags))) {
      signalPassFailure();
      return;
    }

    module.walk([&](Operation *op) {
      if (isLinalgCompute(op) &&
          op->getName().getStringRef() == "linalg.matmul")
        hadFailure |= !setMxuZero(op);
    });
    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
