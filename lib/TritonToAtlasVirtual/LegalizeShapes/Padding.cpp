// Pass 4, padding: neutral values, padding a linalg op up to whole Atlas
// tiles, and the per-tile ttatlas.pad tags at the memory boundary.
#include "LegalizeShapes.h"

#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
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

// A pad tag's shape is always [rows, cols]: a row of n lanes is [1, n].
static SmallVector<int64_t> rowsAndColumns(ArrayRef<int64_t> shape) {
  if (shape.size() == 1)
    return {1, shape.front()};
  return SmallVector<int64_t>(shape);
}

std::optional<ttatlas::PadNeutral> constantNeutral(Value value) {
  // A tile-wide constant is a scalar constant behind splats and broadcasts.
  if (Operation *definition = value.getDefiningOp())
    if (isa<SplatOp, BroadcastOp>(definition))
      return constantNeutral(definition->getOperand(0));
  if (matchPattern(value, m_AnyZeroFloat()) || matchPattern(value, m_Zero()))
    return ttatlas::PadNeutral::zero;
  if (matchPattern(value, m_OneFloat()) || matchPattern(value, m_One()))
    return ttatlas::PadNeutral::one;
  if (matchPattern(value, m_NegInfFloat()))
    return ttatlas::PadNeutral::neg_inf;
  if (matchPattern(value, m_PosInfFloat()))
    return ttatlas::PadNeutral::pos_inf;
  return std::nullopt;
}

// The enum value of a reduction's identity, as arith::getNeutralElement
// returns it (0, 1, -inf or +inf); anything else has no host fill.
static std::optional<ttatlas::PadNeutral> neutralFromAttr(TypedAttr attr) {
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    if (integer.getValue().isZero())
      return ttatlas::PadNeutral::zero;
    if (integer.getValue().isOne())
      return ttatlas::PadNeutral::one;
    return std::nullopt;
  }
  if (auto floating = dyn_cast<FloatAttr>(attr)) {
    const llvm::APFloat &number = floating.getValue();
    if (number.isZero())
      return ttatlas::PadNeutral::zero;
    if (number.isExactlyValue(1.0))
      return ttatlas::PadNeutral::one;
    if (number.isInfinity())
      return number.isNegative() ? ttatlas::PadNeutral::neg_inf
                                 : ttatlas::PadNeutral::pos_inf;
  }
  return std::nullopt;
}

// A padded lane only ever reaches a result through a reduced dimension: the
// reduction's combiner reads it. Padding along a parallel dimension is cropped
// later, so it can hold anything. So the pad value is the identity of the
// combiner of every reduction over a padded dimension, and zero (the cheapest
// host fill) when there is none. `padded[i]` says whether loop i is padded; an
// empty list means all of them.
std::optional<ttatlas::PadNeutral>
operationNeutral(Operation *op, ArrayRef<bool> padded, std::string *why) {
  auto fail = [&](const Twine &reason) -> std::optional<ttatlas::PadNeutral> {
    if (why)
      *why = reason.str();
    return std::nullopt;
  };
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return fail("not a linalg operation");
  // A matmul reduces over K, and its padded K lanes must add nothing.
  if (linalg::isaContractionOpInterface(linalgOp))
    return ttatlas::PadNeutral::zero;

  SmallVector<utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
  auto isPaddedReduction = [&](unsigned loop) {
    return iterators[loop] == utils::IteratorType::reduction &&
           (padded.empty() || padded[loop]);
  };
  SmallVector<BlockArgument> accumulators(
      linalgOp.getRegionOutputArgs().begin(),
      linalgOp.getRegionOutputArgs().end());

  std::optional<ttatlas::PadNeutral> result;
  for (auto [position, output] : llvm::enumerate(linalgOp.getDpsInits())) {
    // An output is reduced when some reduced loop is missing from its map.
    AffineMap map =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(position));
    bool reduced = false;
    for (unsigned loop = 0; loop < iterators.size(); ++loop)
      if (isPaddedReduction(loop) && !map.isFunctionOfDim(loop))
        reduced = true;
    if (!reduced)
      continue;
    SmallVector<Operation *> combiners;
    if (!matchReduction(accumulators, position, combiners) ||
        combiners.size() != 1)
      return fail("its reduction body is not a single combiner op that "
                  "mlir::matchReduction recognizes");
    // MLIR's identity for maxnumf and minnumf is a NaN, which the host cannot
    // fill. -inf and +inf are identities too (maxnumf(x, -inf) = x), so use
    // those.
    std::optional<ttatlas::PadNeutral> neutral;
    if (isa<arith::MaxNumFOp>(combiners.front())) {
      neutral = ttatlas::PadNeutral::neg_inf;
    } else if (isa<arith::MinNumFOp>(combiners.front())) {
      neutral = ttatlas::PadNeutral::pos_inf;
    } else {
      std::optional<TypedAttr> identity =
          arith::getNeutralElement(combiners.front());
      if (!identity)
        return fail("its reduction combiner has no identity element");
      neutral = neutralFromAttr(*identity);
    }
    if (!neutral)
      return fail("its reduction identity is not 0, 1, -inf or +inf, the "
                  "values the host can fill");
    if (result && *result != *neutral)
      return fail("reductions that read the same padded value need different "
                  "pad values; masking inside the kernel before each "
                  "reduction is not implemented yet");
    result = neutral;
  }
  return result.value_or(ttatlas::PadNeutral::zero);
}

// The pad value as an attribute of `type`; null when the type cannot hold it
// (an infinity needs a float type that has one).
static Attribute padValueAttr(ttatlas::PadNeutral neutral, Type type) {
  using ttatlas::PadNeutral;
  if (neutral == PadNeutral::zero)
    return Builder(type.getContext()).getZeroAttr(type);
  if (auto floatType = dyn_cast<FloatType>(type)) {
    const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
    if (neutral == PadNeutral::one)
      return FloatAttr::get(floatType, 1.0);
    if (!llvm::APFloat::semanticsHasInf(semantics))
      return {};
    return FloatAttr::get(
        floatType, llvm::APFloat::getInf(semantics, /*Negative=*/neutral ==
                                                        PadNeutral::neg_inf));
  }
  if (auto integerType = dyn_cast<IntegerType>(type);
      integerType && neutral == PadNeutral::one)
    return IntegerAttr::get(integerType, 1);
  return {};
}

LogicalResult padToAtlasTiles(IRRewriter &rewriter, linalg::LinalgOp op,
                              linalg::LinalgOp &paddedOp,
                              SmallVectorImpl<Value> &replacements) {
  SmallVector<OpFoldResult> paddingSizes(
      op.getNumLoops(), rewriter.getIndexAttr(ttatlas::target::kTileSize));

  // Pad inputs and output init tensors with the pad value that the reductions
  // over padded dimensions need (zero when there is none), so a padded lane
  // cannot change a reduced result.
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  SmallVector<bool> paddedLoops;
  for (int64_t size : ranges)
    paddedLoops.push_back(size % ttatlas::target::kTileSize != 0);
  std::string why;
  std::optional<ttatlas::PadNeutral> neutral =
      operationNeutral(op, paddedLoops, &why);
  if (!neutral) {
    ttatlas::emitUnsupported(
        op, "cannot establish a pad value for static tile padding: " + why);
    return failure();
  }
  SmallVector<Attribute> paddingValues;
  for (Value operand : op->getOperands()) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    Attribute value = padValueAttr(*neutral, elementType);
    if (!value) {
      ttatlas::emitUnsupported(
          op, "the pad value cannot be written in this element type");
      return failure();
    }
    paddingValues.push_back(value);
  }

  auto tilingOp = dyn_cast<TilingInterface>(op.getOperation());
  if (!tilingOp) {
    ttatlas::emitUnsupported(op, "linalg operation does not implement the "
                                 "MLIR TilingInterface");
    return failure();
  }
  linalg::PadTilingInterfaceOptions options;
  options.setPaddingSizes(paddingSizes);
  options.setPadToMultipleOf(true);
  options.setPaddingValues(paddingValues);
  FailureOr<linalg::PadTilingInterfaceResult> padded =
      linalg::rewriteAsPaddedOp(rewriter, tilingOp, options);
  if (failed(padded)) {
    ttatlas::emitUnsupported(op,
                             "MLIR failed to pad linalg iteration dimensions");
    return failure();
  }
  paddedOp = cast<linalg::LinalgOp>(padded->paddedOp.getOperation());
  replacements.assign(padded->replacements);
  SmallVector<tensor::PadOp> padOps = padded->padOps;
  for (tensor::PadOp padOp : padOps) {
    auto sourceType = dyn_cast<RankedTensorType>(padOp.getSource().getType());
    if (!sourceType)
      continue;
    auto pad = ttatlas::PadAttr::get(rewriter.getContext(), *neutral,
                                     rowsAndColumns(sourceType.getShape()));
    padOp->setAttr(ttatlas::kPad, pad);
  }
  if (!padOps.empty())
    ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                "pad static linalg dimensions to 32");
  return success();
}

namespace {

constexpr int64_t kTile = ttatlas::target::kTileSize;

// One boundary tile: the op that yields it (it carries the tile's tag) and
// where the tile sits in the boundary tensor, as [row, col].
struct Tile {
  Operation *op;
  SmallVector<int64_t, 2> offset;
};

// [rows, cols] of one tile of `type`: 32x32, or 1x32 for a row.
SmallVector<int64_t, 2> tileExtent(RankedTensorType type) {
  return {type.getRank() == 1 ? 1 : kTile, kTile};
}

bool isOneTile(RankedTensorType type) {
  return llvm::all_of(type.getShape(),
                      [](int64_t size) { return size == kTile; });
}

bool isWholeTiles(RankedTensorType type) {
  return llvm::all_of(type.getShape(),
                      [](int64_t size) { return size % kTile == 0; });
}

// The value of a slice offset. Unrolling a tile loop leaves offsets such as
// 0 + 32 * 1 as index arithmetic on constants, which only canonicalization
// folds, so evaluate sums and products of constants here.
std::optional<int64_t> constantOffset(OpFoldResult at) {
  if (std::optional<int64_t> constant = getConstantIntValue(at))
    return constant;
  Operation *definition = dyn_cast<Value>(at).getDefiningOp();
  if (!isa_and_present<arith::AddIOp, arith::MulIOp>(definition))
    return std::nullopt;
  std::optional<int64_t> lhs = constantOffset(definition->getOperand(0));
  std::optional<int64_t> rhs = constantOffset(definition->getOperand(1));
  if (!lhs || !rhs)
    return std::nullopt;
  return isa<arith::AddIOp>(definition) ? *lhs + *rhs : *lhs * *rhs;
}

// Where a slice cuts exactly one tile at constant offsets, with unit strides,
// as [row, col]: what tiling leaves. The offsets may still be index constants
// that canonicalization has not folded into the slice yet.
template <typename SliceOp>
std::optional<SmallVector<int64_t, 2>> tileSliceOffset(SliceOp slice) {
  if (!llvm::all_of(slice.getStaticSizes(),
                    [](int64_t size) { return size == kTile; }) ||
      !llvm::all_of(slice.getMixedStrides(), [](OpFoldResult stride) {
        return getConstantIntValue(stride) == 1;
      }))
    return std::nullopt;
  SmallVector<int64_t, 2> offset;
  for (OpFoldResult at : slice.getMixedOffsets()) {
    std::optional<int64_t> constant = constantOffset(at);
    if (!constant)
      return std::nullopt;
    offset.push_back(*constant);
  }
  if (offset.size() == 1)
    offset.insert(offset.begin(), 0);
  return offset;
}

// A slice with its offsets written as static numbers. Canonicalization does
// that rewrite itself, but into a new op that drops the tile's tag, so it is
// done here, before the tag is set. Slices that are static already are kept.
tensor::ExtractSliceOp withStaticOffsets(RewriterBase &rewriter,
                                         tensor::ExtractSliceOp slice,
                                         ArrayRef<int64_t> offset) {
  if (!llvm::any_of(slice.getStaticOffsets(), ShapedType::isDynamic))
    return slice;
  SmallVector<OpFoldResult> offsets;
  for (int64_t at :
       ArrayRef<int64_t>(offset).take_back(slice.getSourceType().getRank()))
    offsets.push_back(rewriter.getIndexAttr(at));
  rewriter.setInsertionPoint(slice);
  auto fresh = tensor::ExtractSliceOp::create(
      rewriter, slice.getLoc(), slice.getType(), slice.getSource(), offsets,
      slice.getMixedSizes(), slice.getMixedStrides());
  fresh->setDiscardableAttrs(slice->getDiscardableAttrDictionary());
  rewriter.replaceOp(slice, fresh);
  return fresh;
}

// Sets `pad` on `op`. Two different tags cannot both describe the one tile
// that `op` yields.
LogicalResult setTag(Operation *op, ttatlas::PadAttr pad) {
  if (Attribute old = op->getAttr(ttatlas::kPad); old && old != pad) {
    ttatlas::emitUnsupported(
        op, "conflicting ttatlas.pad metadata: the op yields tiles with "
            "different padding");
    return failure();
  }
  op->setAttr(ttatlas::kPad, pad);
  return success();
}

// The tiles of one boundary tensor, and what is real in it.
struct Boundary {
  SmallVector<Tile> tiles;
  SmallVector<int64_t, 2> extent;
  SmallVector<int64_t> logical; // [rows, cols] of the real lanes
  ttatlas::PadNeutral neutral;
  bool droppable; // a tile with no real lane may be erased if unread (loads)
};

// Gives each tile that holds padded lanes its own tag: the part of the logical
// shape that lies inside it. A full tile has no tag. A tile with no real lane
// is erased when it is droppable and nothing reads it, and rejected otherwise:
// the host has nothing to put in it or take from it.
LogicalResult tagTiles(Operation *memory, const Boundary &boundary) {
  for (const Tile &tile : boundary.tiles) {
    SmallVector<int64_t, 2> real;
    bool full = true, empty = false;
    for (unsigned axis = 0; axis < 2; ++axis) {
      real.push_back(std::min(boundary.extent[axis],
                              boundary.logical[axis] - tile.offset[axis]));
      empty |= real.back() <= 0;
      full &= real.back() == boundary.extent[axis];
    }
    if (empty) {
      if (boundary.droppable && tile.op && tile.op->use_empty()) {
        tile.op->erase();
        continue;
      }
      ttatlas::emitUnsupported(
          memory, "the mask leaves a whole tile without a real lane, and that "
                  "tile is still read or written; use a block that ends at "
                  "the mask's bound");
      return failure();
    }
    if (full)
      continue;
    if (!tile.op) {
      ttatlas::emitUnsupported(
          memory, "a padded tile has no op that could carry its ttatlas.pad "
                  "tag");
      return failure();
    }
    if (failed(setTag(tile.op, ttatlas::PadAttr::get(memory->getContext(),
                                                     boundary.neutral, real))))
      return failure();
  }
  return success();
}

// The tiles of one boundary tensor of a load: the tensor the host fills, which
// is the load itself, or `pad`, the tensor.pad that grows a load that is not
// whole tiles (padToAtlasTiles made it). Its tiles are the tensor itself when
// it is one tile, else its constant tile slices.
LogicalResult tagLoadBoundary(RewriterBase &rewriter, LoadOp load,
                              tensor::PadOp pad, const MaskBox *box) {
  auto type = cast<RankedTensorType>(load.getType());
  Boundary boundary{
      {},
      tileExtent(type),
      rowsAndColumns(box ? ArrayRef<int64_t>(box->bounds) : type.getShape()),
      box ? box->neutral : ttatlas::PadNeutral::zero,
      /*droppable=*/true};
  // A load without a mask box has no padding of its own: it is tagged only
  // when padToAtlasTiles put a pad under it. With a box, failing to find the
  // tiles is an error, since the mask's padding would be lost.
  auto notFound = [&](const Twine &why) {
    if (!box)
      return success();
    ttatlas::emitUnsupported(load, why);
    return failure();
  };

  Value value = pad ? pad.getResult() : load.getResult();
  if (pad && !box)
    boundary.neutral =
        pad->getAttrOfType<ttatlas::PadAttr>(ttatlas::kPad).getNeutral();
  auto valueType = cast<RankedTensorType>(value.getType());
  if (!isWholeTiles(valueType))
    return notFound("a masked load of a partial tile is not read through a "
                    "padding tensor.pad");
  if (isOneTile(valueType)) {
    boundary.tiles.push_back(
        {pad ? pad.getOperation() : load.getOperation(), {0, 0}});
  } else {
    SmallVector<Operation *> users(value.getUsers());
    for (Operation *user : users) {
      auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
      std::optional<SmallVector<int64_t, 2>> offset =
          slice ? tileSliceOffset(slice) : std::nullopt;
      if (!offset)
        return notFound("a masked load must be read only through constant "
                        "tile slices");
      boundary.tiles.push_back(
          {withStaticOffsets(rewriter, slice, *offset), *offset});
    }
  }
  // The pad's own tag describes its whole result; each tile gets its own.
  if (pad)
    pad->removeAttr(ttatlas::kPad);
  return tagTiles(load, boundary);
}

// A load's tiles. When every reader is a tagged tensor.pad of it, each pad is
// a boundary (padding one operand of several linalg ops makes one pad per use,
// which CSE merges later); otherwise the load is.
LogicalResult tagLoadTiles(RewriterBase &rewriter, LoadOp load,
                           const MaskBox *box) {
  auto type = dyn_cast<RankedTensorType>(load.getType());
  if (!type || type.getRank() > 2)
    return success();
  SmallVector<tensor::PadOp> pads;
  for (Operation *user : load->getUsers())
    if (auto pad = dyn_cast<tensor::PadOp>(user);
        pad && pad.getSource() == load.getResult() &&
        pad->hasAttr(ttatlas::kPad))
      pads.push_back(pad);
  if (pads.empty() || pads.size() != llvm::range_size(load->getUsers()))
    return tagLoadBoundary(rewriter, load, nullptr, box);
  for (tensor::PadOp pad : pads)
    if (failed(tagLoadBoundary(rewriter, load, pad, box)))
      return failure();
  return success();
}

// A stored tensor's tiles: the producer of each tile, found through the
// constant tile insert_slices that assemble it. A value that is a tail crop of
// such a tensor (a constant slice from the origin, which is what padding a
// small result leaves) is stored as its real lanes only, so the crop's size
// also bounds the logical shape.
LogicalResult tagStoreTiles(StoreOp store, const MaskBox *box) {
  auto type = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!type || type.getRank() > 2)
    return success();
  Boundary boundary{
      {},
      tileExtent(type),
      rowsAndColumns(box ? ArrayRef<int64_t>(box->bounds) : type.getShape()),
      box ? box->neutral : ttatlas::PadNeutral::zero,
      /*droppable=*/false};
  Value value = store.getValue();
  if (auto crop = value.getDefiningOp<tensor::ExtractSliceOp>();
      crop &&
      llvm::all_of(crop.getMixedOffsets(),
                   [](OpFoldResult at) { return isZeroInteger(at); }) &&
      llvm::all_of(crop.getMixedStrides(),
                   [](OpFoldResult stride) { return isOneInteger(stride); }) &&
      !llvm::any_of(crop.getStaticSizes(), ShapedType::isDynamic)) {
    SmallVector<int64_t> kept = rowsAndColumns(crop.getStaticSizes());
    for (unsigned axis = 0; axis < 2; ++axis)
      boundary.logical[axis] = std::min(boundary.logical[axis], kept[axis]);
    value = crop.getSource();
  }
  auto valueType = cast<RankedTensorType>(value.getType());
  bool built = isWholeTiles(valueType);
  if (built && isOneTile(valueType)) {
    boundary.tiles.push_back({value.getDefiningOp(), {0, 0}});
  } else if (built) {
    while (auto insert = value.getDefiningOp<tensor::InsertSliceOp>()) {
      std::optional<SmallVector<int64_t, 2>> offset = tileSliceOffset(insert);
      if (!offset) {
        built = false;
        break;
      }
      boundary.tiles.push_back({insert.getSource().getDefiningOp(), *offset});
      value = insert.getDest();
    }
  }
  if (!built) {
    // Padding with no tile to describe it would be lost.
    if (llvm::equal(boundary.logical, rowsAndColumns(valueType.getShape())))
      return success();
    ttatlas::emitUnsupported(
        store, "a padded stored value must be one tile or built from tiles "
               "(constant tile insert_slices), so its padding can be tagged "
               "per tile");
    return failure();
  }
  return tagTiles(store, boundary);
}

} // namespace

LogicalResult tagBoundaryTiles(ModuleOp module, const MaskBoxes &boxes) {
  SmallVector<Operation *> accesses;
  module.walk([&](Operation *op) {
    if (isa<LoadOp, StoreOp>(op))
      accesses.push_back(op);
  });
  IRRewriter rewriter(module.getContext());
  for (Operation *op : accesses) {
    auto box = boxes.find(op);
    const MaskBox *mask = box == boxes.end() ? nullptr : &box->second;
    if (failed(isa<LoadOp>(op) ? tagLoadTiles(rewriter, cast<LoadOp>(op), mask)
                               : tagStoreTiles(cast<StoreOp>(op), mask)))
      return failure();
  }
  return success();
}

} // namespace mlir::triton
