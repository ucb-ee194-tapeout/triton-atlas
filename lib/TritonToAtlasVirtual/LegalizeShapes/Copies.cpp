// Pass 4, plain copies: a tt.store of a tt.load with nothing computed between
// them. Padding and tiling work on linalg ops, so a copy has none to work on;
// this puts the boundary forms the stage-4 contract asks for around it.
#include "LegalizeShapes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton {

namespace {

constexpr int64_t kTile = ttatlas::target::kTileSize;

bool isOneTileShape(ArrayRef<int64_t> shape) {
  return llvm::all_of(shape, [](int64_t size) { return size == kTile; });
}

} // namespace

LogicalResult tileBareCopies(ModuleOp module) {
  SmallVector<StoreOp> copies;
  module.walk([&](StoreOp store) {
    auto type = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (type && type.getRank() <= 2 && type.hasStaticShape() &&
        store.getValue().getDefiningOp<LoadOp>() &&
        !isOneTileShape(type.getShape()))
      copies.push_back(store);
  });
  IRRewriter rewriter(module.getContext());
  for (StoreOp store : copies) {
    auto type = cast<RankedTensorType>(store.getValue().getType());
    ArrayRef<int64_t> shape = type.getShape();
    SmallVector<int64_t> whole;
    for (int64_t size : shape)
      whole.push_back(llvm::divideCeil(size, kTile) * kTile);
    auto wholeType = RankedTensorType::get(whole, type.getElementType());
    Location loc = store.getLoc();
    rewriter.setInsertionPoint(store);

    // A load that is not whole tiles grows to whole tiles through one
    // tensor.pad; the host fills its padded lanes from the pad's tag (the tag
    // itself is set per tile, once the tiles exist).
    Value value = store.getValue();
    if (wholeType != type) {
      Value zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(type.getElementType()));
      auto pad = tensor::createPadHighOp(wholeType, value, zero,
                                         /*nofold=*/false, loc, rewriter);
      SmallVector<int64_t> rowsCols(shape);
      if (rowsCols.size() == 1)
        rowsCols.insert(rowsCols.begin(), 1);
      pad->setAttr(ttatlas::kPad,
                   ttatlas::PadAttr::get(rewriter.getContext(),
                                         ttatlas::PadNeutral::zero, rowsCols));
      value = pad.getResult();
    }

    // Cut into tiles and put the tiles back together, over a fresh empty.
    if (!isOneTileShape(whole)) {
      Value assembled =
          tensor::EmptyOp::create(rewriter, loc, whole, type.getElementType());
      SmallVector<int64_t> tileShape(whole.size(), kTile);
      SmallVector<OpFoldResult> sizes(whole.size(),
                                      rewriter.getIndexAttr(kTile));
      SmallVector<OpFoldResult> strides(whole.size(), rewriter.getIndexAttr(1));
      for (int64_t row = 0; row < whole.front(); row += kTile) {
        for (int64_t col = 0; col < (whole.size() == 2 ? whole.back() : 1);
             col += kTile) {
          SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(row)};
          if (whole.size() == 2)
            offsets.push_back(rewriter.getIndexAttr(col));
          Value tile = tensor::ExtractSliceOp::create(rewriter, loc, value,
                                                      offsets, sizes, strides);
          assembled = tensor::InsertSliceOp::create(
              rewriter, loc, tile, assembled, offsets, sizes, strides);
        }
      }
      value = assembled;
    }

    // The store takes the real lanes only: a crop from the origin.
    if (wholeType != type) {
      SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      for (int64_t size : shape)
        sizes.push_back(rewriter.getIndexAttr(size));
      SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
      value = tensor::ExtractSliceOp::create(rewriter, loc, type, value,
                                             offsets, sizes, strides);
    }
    rewriter.modifyOpInPlace(store, [&] { store->setOperand(1, value); });
    ttatlas::recordLegalization(store, "ttatlas-legalize-shapes",
                                "tile a plain copy");
  }
  return success();
}

} // namespace mlir::triton
