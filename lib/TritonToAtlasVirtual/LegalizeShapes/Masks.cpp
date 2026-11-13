// Pass 4, mask analysis: turns the bounds a load or store mask proves into
// ttatlas.pad tags (the neutral value of the padded lanes).
#include "LegalizeShapes.h"

#include "TritonToAtlasVirtual/LaneValues.h"
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

static std::optional<ttatlas::PadNeutral> requiredNeutral(Value value,
                                                          unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "arith.truncf" || name == "arith.extf" ||
       name == "tt.fp_to_fp") &&
      definition->getNumOperands() == 1)
    return requiredNeutral(definition->getOperand(0), depth + 1);
  if (name == "tensor.extract_slice" && definition->getNumResults() == 1) {
    std::optional<ttatlas::PadNeutral> result;
    for (Operation *user : definition->getResult(0).getUsers()) {
      std::optional<ttatlas::PadNeutral> current =
          user->getName().getStringRef() == "tensor.extract_slice"
              ? requiredNeutral(user->getResult(0), depth + 1)
              : operationNeutral(user);
      if (!current || (result && *result != *current))
        return std::nullopt;
      result = current;
    }
    return result;
  }
  if (name == "tensor.insert_slice" && definition->getNumOperands() > 0)
    return requiredNeutral(definition->getOperand(0), depth + 1);
  if (name == "tt.load") {
    std::optional<ttatlas::PadNeutral> result;
    for (Operation *user : value.getUsers()) {
      std::optional<ttatlas::PadNeutral> current =
          user->getName().getStringRef() == "tensor.extract_slice"
              ? requiredNeutral(user->getResult(0), depth + 1)
              : operationNeutral(user);
      if (!current || (result && *result != *current))
        return std::nullopt;
      result = current;
    }
    return result;
  }
  return operationNeutral(definition);
}

LogicalResult
tagMaskedBoundaries(ModuleOp module,
                    DenseMap<Operation *, Attribute> &storePadTags) {
  bool failed = false;
  DenseMap<Operation *, Attribute> tags;
  module.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    bool isLoad = name == "tt.load";
    bool isStore = name == "tt.store";
    if ((!isLoad && !isStore) || op->getNumOperands() < (isLoad ? 2u : 3u))
      return;
    Value mask = op->getOperand(isLoad ? 1 : 2);
    Value value = isLoad ? op->getResult(0) : op->getOperand(1);
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || (tensor.getRank() != 1 && tensor.getRank() != 2))
      return;
    // The shared lane evaluator decides what "compile-time mask" means: it
    // follows expand_dims/broadcast axes and constant bounds. The true lanes
    // must form the leading box of the tile; that box is the logical shape.
    std::optional<SmallVector<int64_t>> bounds = maskBounds(mask);
    if (!bounds || bounds->size() != tensor.getShape().size()) {
      ttatlas::emitUnsupported(
          op,
          "edge mask is not a static make_range comparison against a constant: "
          "its true lanes are not a compile-time leading box (a runtime "
          "bound, or a mask like offs > k)");
      failed = true;
      return;
    }
    if (llvm::is_contained(*bounds, 0)) {
      ttatlas::emitUnsupported(op, "edge mask has no true lane");
      failed = true;
      return;
    }
    SmallVector<int64_t> logicalShape = *bounds;
    if (logicalShape == tensor.getShape())
      return;

    std::optional<ttatlas::PadNeutral> neutral;
    if (isLoad) {
      if (op->getNumOperands() < 3) {
        ttatlas::emitUnsupported(
            op, "masked load has no explicit neutral value for padded lanes");
        failed = true;
        return;
      }
      neutral = constantNeutral(op->getOperand(2));
    }
    if (isStore)
      neutral = requiredNeutral(value);
    if (!neutral) {
      ttatlas::emitUnsupported(
          op, "masked padded value has no supported neutral element");
      failed = true;
      return;
    }
    if (isLoad) {
      std::optional<ttatlas::PadNeutral> consumerNeutral =
          requiredNeutral(value);
      if (!consumerNeutral || *consumerNeutral != *neutral) {
        ttatlas::emitUnsupported(
            op, "masked load neutral element does not match its consumer");
        failed = true;
        return;
      }
    }
    Operation *taggedOp = isLoad ? op : value.getDefiningOp();
    if (!taggedOp) {
      ttatlas::emitUnsupported(op,
                               "padded store value has no defining operation");
      failed = true;
      return;
    }
    auto pad = ttatlas::PadAttr::get(op->getContext(), *neutral, logicalShape);
    if (Attribute old = taggedOp->getAttr(ttatlas::kPad); old && old != pad) {
      ttatlas::emitUnsupported(
          taggedOp, "conflicting ttatlas.pad metadata from edge masks");
      failed = true;
      return;
    }
    auto [it, inserted] = tags.try_emplace(taggedOp, pad);
    if (!inserted && it->second != pad) {
      ttatlas::emitUnsupported(
          taggedOp, "conflicting ttatlas.pad metadata from edge masks");
      failed = true;
    }
  });
  if (failed)
    return failure();
  for (auto [op, pad] : tags) {
    op->setAttr(ttatlas::kPad, pad);
    ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                "recognize compile-time edge mask");
  }
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tt.store" &&
        op->getNumOperands() >= 3) {
      Value value = op->getOperand(1);
      if (Operation *definition = value.getDefiningOp())
        if (Attribute pad = definition->getAttr(ttatlas::kPad))
          storePadTags.try_emplace(op, pad);
    }
  });
  return success();
}

} // namespace mlir::triton
