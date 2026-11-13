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

    // The host's fill of a load's padded lanes: the identity of the reductions
    // that read them unchanged, else `other` when it is a fill the host can
    // write, else zero. A store's padded lanes are cropped by the host, never
    // written, so its value needs no neutral (zero).
    ttatlas::PadNeutral neutral = ttatlas::PadNeutral::zero;
    if (load) {
      if (!load.getOther()) {
        ttatlas::emitUnsupported(
            op, "masked load has no explicit neutral value for padded lanes");
        failed = true;
        return;
      }
      SmallVector<bool> padded;
      for (auto [bound, size] : llvm::zip_equal(*bounds, tensor.getShape()))
        padded.push_back(bound < size);
      if (std::optional<ttatlas::PadNeutral> fill = hostFill(value, padded))
        neutral = *fill;
      else if (std::optional<ttatlas::PadNeutral> other =
                   constantNeutral(load.getOther()))
        neutral = *other;
    }
    boxes[op] = {neutral, *bounds};
    ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                "recognize compile-time edge mask");
  });
  return failure(failed);
}

} // namespace mlir::triton
