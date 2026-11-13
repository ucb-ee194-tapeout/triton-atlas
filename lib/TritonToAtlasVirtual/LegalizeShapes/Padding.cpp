// Pass 4, padding: neutral values, padding a linalg op up to whole Atlas
// tiles, and carrying a store's pad tag to the op that produces its value.
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
                                     sourceType.getShape());
    padOp->setAttr(ttatlas::kPad, pad);
  }
  if (!padOps.empty())
    ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                "pad static linalg dimensions to 32");
  return success();
}

LogicalResult
propagateStorePadTags(const DenseMap<Operation *, Attribute> &storePadTags) {
  for (auto [store, pad] : storePadTags) {
    Value storedValue = store->getOperand(1);
    Operation *definition = storedValue.getDefiningOp();
    if (!definition) {
      ttatlas::emitUnsupported(
          store, "padded store value has no defining operation after tiling");
      return failure();
    }
    if (Attribute old = definition->getAttr(ttatlas::kPad); old && old != pad) {
      ttatlas::emitUnsupported(definition,
                               "conflicting ttatlas.pad metadata after tiling");
      return failure();
    }
    definition->setAttr(ttatlas::kPad, pad);
  }
  return success();
}

} // namespace mlir::triton
