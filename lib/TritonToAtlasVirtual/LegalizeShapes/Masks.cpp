// Pass 4, mask analysis: reads the bounds a load or store mask proves and the
// neutral value of the padded lanes. Padding.cpp turns them into per-tile
// ttatlas.pad tags once tiling has made the tiles.
#include "LegalizeShapes.h"

#include "TritonToAtlasVirtual/LaneValues.h"
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace mlir::triton {

// The pad value the consumers of `value` need in its padded lanes, looking
// through conversions and slices to the ops that read it.
static std::optional<ttatlas::PadNeutral> requiredNeutral(Value value,
                                                          unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  // What the readers of `value` need: a slice passes the question on to its
  // own readers, anything else answers for itself.
  auto readersNeed = [&](Value read) -> std::optional<ttatlas::PadNeutral> {
    std::optional<ttatlas::PadNeutral> result;
    for (Operation *user : read.getUsers()) {
      std::optional<ttatlas::PadNeutral> current =
          isa<tensor::ExtractSliceOp>(user)
              ? requiredNeutral(user->getResult(0), depth + 1)
              : operationNeutral(user);
      if (!current || (result && *result != *current))
        return std::nullopt;
      result = current;
    }
    return result;
  };
  if (isa<arith::TruncFOp, arith::ExtFOp, FpToFpOp>(definition))
    return requiredNeutral(definition->getOperand(0), depth + 1);
  if (isa<tensor::ExtractSliceOp>(definition))
    return readersNeed(definition->getResult(0));
  if (isa<tensor::InsertSliceOp>(definition))
    return requiredNeutral(definition->getOperand(0), depth + 1);
  if (isa<LoadOp>(definition))
    return readersNeed(value);
  return operationNeutral(definition);
}

LogicalResult readMaskedBoundaries(ModuleOp module, MaskBoxes &boxes) {
  bool failed = false;
  module.walk([&](Operation *op) {
    auto load = dyn_cast<LoadOp>(op);
    auto store = dyn_cast<StoreOp>(op);
    if (!load && !store)
      return;
    Value mask = load ? load.getMask() : store.getMask();
    if (!mask)
      return;
    Value value = load ? load.getResult() : store.getValue();
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || (tensor.getRank() != 1 && tensor.getRank() != 2))
      return;
    // The shared lane evaluator decides what "compile-time mask" means: it
    // follows expand_dims/broadcast axes and constant bounds. The true lanes
    // must form the leading box of the tile; that box is the logical shape.
    std::optional<SmallVector<int64_t>> bounds = maskBounds(mask);
    if (!bounds || bounds->size() != tensor.getShape().size()) {
      ttatlas::emitUnsupported(
          op, "edge mask is not a compile-time leading box (a runtime bound, "
              "or a mask like offs > k)");
      failed = true;
      return;
    }
    if (llvm::is_contained(*bounds, 0)) {
      ttatlas::emitUnsupported(op, "edge mask has no true lane");
      failed = true;
      return;
    }
    if (ArrayRef<int64_t>(*bounds) == tensor.getShape())
      return;

    std::optional<ttatlas::PadNeutral> neutral;
    if (load) {
      if (!load.getOther()) {
        ttatlas::emitUnsupported(
            op, "masked load has no explicit neutral value for padded lanes");
        failed = true;
        return;
      }
      neutral = constantNeutral(load.getOther());
    } else {
      neutral = requiredNeutral(value);
    }
    if (!neutral) {
      ttatlas::emitUnsupported(
          op, "masked padded value has no supported neutral element");
      failed = true;
      return;
    }
    if (load) {
      std::optional<ttatlas::PadNeutral> consumerNeutral =
          requiredNeutral(value);
      if (!consumerNeutral || *consumerNeutral != *neutral) {
        ttatlas::emitUnsupported(
            op, "masked load neutral element does not match its consumer");
        failed = true;
        return;
      }
    }
    boxes[op] = {*neutral, *bounds};
    ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                "recognize compile-time edge mask");
  });
  return failure(failed);
}

} // namespace mlir::triton
