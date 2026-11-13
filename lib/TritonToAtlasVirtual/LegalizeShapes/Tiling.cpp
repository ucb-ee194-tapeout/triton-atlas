// Pass 4, tiling: MLIR's SCF tile-and-fuse driver helpers, the MXU policy tag,
// tile-loop unrolling and fresh fill destinations.
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

// True when `op` carries a ttatlas.mxu other than 0; reports it. A matmul must
// stay on MXU0 because MXU1 cannot run a K chain.
bool hasConflictingMxu(Operation *op) {
  Attribute oldUnit = op->getAttr(ttatlas::kMxu);
  if (!oldUnit)
    return false;
  auto oldInteger = dyn_cast<IntegerAttr>(oldUnit);
  if (oldInteger && oldInteger.getType().isSignlessInteger(32) &&
      oldInteger.getInt() == 0)
    return false;
  ttatlas::emitUnsupported(
      op, "linalg.matmul has conflicting ttatlas.mxu; expected 0");
  return true;
}

bool setMxuZero(Operation *op) {
  if (hasConflictingMxu(op))
    return false;
  // atlas-hw: MXU0 is the pass 4 default policy; MXU1 cannot run a K chain.
  op->setAttr(ttatlas::kMxu,
              IntegerAttr::get(IntegerType::get(op->getContext(), 32), 0));
  return true;
}

// Tile sizes that cut every dimension larger than one Atlas tile down to one
// tile and leave smaller ones whole (0 means "do not tile this loop"). With
// `parallelOnly`, reduction loops are left whole: producers (a fill feeding the
// accumulator) can only be fused into a loop nest that does not iterate over
// the reduction, or they would re-initialize the accumulator on every step.
SmallVector<OpFoldResult> atlasTileSizes(Builder &builder, linalg::LinalgOp op,
                                         bool parallelOnly) {
  SmallVector<OpFoldResult> sizes;
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  for (auto [range, iterator] :
       llvm::zip_equal(op.getStaticLoopRanges(), iterators)) {
    bool tile = range > ttatlas::target::kTileSize &&
                !(parallelOnly && iterator == utils::IteratorType::reduction);
    sizes.push_back(
        builder.getIndexAttr(tile ? ttatlas::target::kTileSize : 0));
  }
  return sizes;
}

bool exceedsTile(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp &&
         llvm::any_of(linalgOp.getStaticLoopRanges(),
                      [](int64_t n) { return n > ttatlas::target::kTileSize; });
}

// Fuse only linalg producers whose loops are whole tiles. Other
// tiling-interface producers, such as the tensor.pad that carries a ttatlas.pad
// tag, must stay whole. A producer with a loop that is not a multiple of the
// tile (a K of 16 under a 64x32 consumer) is left out too: a fused clone is
// only ever tiled, never padded, so it would stay short of a tile. Left
// whole, the driver pads it on its own turn and the consumer reads slices of
// the padded result.
std::optional<scf::SCFTileAndFuseOptions::ControlFnResult>
fuseLinalgProducersOnly(tensor::ExtractSliceOp, OpResult producer, bool) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(producer.getOwner());
  if (!linalgOp)
    return std::nullopt;
  if (llvm::any_of(linalgOp.getStaticLoopRanges(), [](int64_t size) {
        return size % ttatlas::target::kTileSize != 0;
      }))
    return std::nullopt;
  return scf::SCFTileAndFuseOptions::ControlFnResult{};
}

// Fully unroll every tile loop. The walk is post-order, so an inner loop is
// expanded before the outer loop that is about to clone it.
LogicalResult unrollTileLoops(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    if (failed(loopUnrollFull(loop))) {
      ttatlas::emitUnsupported(
          loop, "tile loop does not have a statically known trip count");
      return failure();
    }
  }
  return success();
}

// A linalg.fill overwrites its whole destination, so what the destination held
// before is dead. Tiling and unrolling leave a tile fill writing a slice of
// the previous tile's result; hand it a fresh tensor.empty tile instead, which
// is what the stage-4 contract (and later passes) expect.
void freshFillDestinations(ModuleOp module, IRRewriter &rewriter) {
  SmallVector<linalg::FillOp> fills;
  module.walk([&](linalg::FillOp fill) {
    if (!fill.getOutputs().front().getDefiningOp<tensor::EmptyOp>())
      fills.push_back(fill);
  });
  for (linalg::FillOp fill : fills) {
    auto type = dyn_cast<RankedTensorType>(fill.getOutputs().front().getType());
    if (!type)
      continue;
    rewriter.setInsertionPoint(fill);
    auto empty = tensor::EmptyOp::create(
        rewriter, fill.getLoc(), type.getShape(), type.getElementType());
    rewriter.modifyOpInPlace(
        fill, [&] { fill.getOutputsMutable()[0].set(empty.getResult()); });
  }
}

} // namespace mlir::triton
