// Pass 4, padding: neutral values, padding a linalg op up to whole Atlas
// tiles, and carrying a store's pad tag to the op that produces its value.
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

std::optional<ttatlas::PadNeutral> constantNeutral(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "tt.splat" || name == "tt.broadcast") &&
      definition->getNumOperands() == 1)
    return constantNeutral(definition->getOperand(0));
  if (name != "arith.constant")
    return std::nullopt;
  Attribute attribute = definition->getAttr("value");
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(attribute)) {
    if (!dense.isSplat())
      return std::nullopt;
    if (isa<FloatType>(dense.getElementType())) {
      const llvm::APFloat &number = dense.getSplatValue<llvm::APFloat>();
      if (number.isZero())
        return ttatlas::PadNeutral::zero;
      if (number.isInfinity() && number.isNegative())
        return ttatlas::PadNeutral::neg_inf;
      return std::nullopt;
    }
    if (isa<IntegerType>(dense.getElementType()))
      return dense.getSplatValue<llvm::APInt>().isZero()
                 ? std::optional(ttatlas::PadNeutral::zero)
                 : std::nullopt;
    return std::nullopt;
  }
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    return integer.getInt() == 0 ? std::optional(ttatlas::PadNeutral::zero)
                                 : std::nullopt;
  if (auto floating = dyn_cast_or_null<FloatAttr>(attribute)) {
    const llvm::APFloat &number = floating.getValue();
    if (number.isZero())
      return ttatlas::PadNeutral::zero;
    if (number.isInfinity() && number.isNegative())
      return ttatlas::PadNeutral::neg_inf;
  }
  return std::nullopt;
}

std::optional<ttatlas::PadNeutral> operationNeutral(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "linalg.matmul" || name == "linalg.transpose" ||
      name == "linalg.broadcast")
    return ttatlas::PadNeutral::zero;
  if (name == "linalg.fill" && op->getNumOperands() > 0)
    return constantNeutral(op->getOperand(0));

  bool reduction = name == "linalg.reduce";
  std::optional<ttatlas::PadNeutral> inferred;
  op->walk([&](Operation *nested) {
    StringRef nestedName = nested->getName().getStringRef();
    if (nestedName == "arith.addf" || nestedName == "arith.addi" ||
        nestedName == "arith.subf" || nestedName == "arith.mulf" ||
        nestedName == "arith.minnumf" || nestedName == "arith.maxnumf")
      inferred = ttatlas::PadNeutral::zero;
    if (nestedName == "arith.maximumf" || nestedName == "arith.maxnumf" ||
        nestedName == "arith.maxsi" || nestedName == "arith.maxui")
      inferred =
          reduction ? ttatlas::PadNeutral::neg_inf : ttatlas::PadNeutral::zero;
  });
  return inferred;
}

LogicalResult padToAtlasTiles(IRRewriter &rewriter, linalg::LinalgOp op,
                              linalg::LinalgOp &paddedOp,
                              SmallVectorImpl<Value> &replacements) {
  SmallVector<OpFoldResult> paddingSizes(
      op.getNumLoops(), rewriter.getIndexAttr(ttatlas::target::kTileSize));

  // Pad inputs and output init tensors with the operation's neutral element.
  // Max reductions use -inf so values introduced by padding cannot win.
  std::optional<ttatlas::PadNeutral> neutral = operationNeutral(op);
  if (!neutral) {
    ttatlas::emitUnsupported(
        op, "cannot establish a neutral element for static tile padding");
    return failure();
  }
  SmallVector<Attribute> paddingValues;
  for (Value operand : op->getOperands()) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    Attribute value;
    if (*neutral == ttatlas::PadNeutral::neg_inf) {
      auto floatType = dyn_cast<FloatType>(elementType);
      if (!floatType) {
        ttatlas::emitUnsupported(op,
                                 "negative-infinity padding requires floats");
        return failure();
      }
      value = FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/true));
    } else {
      value = rewriter.getZeroAttr(elementType);
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
