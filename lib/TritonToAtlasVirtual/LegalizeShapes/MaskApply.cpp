// Pass 4, padded lanes and reductions. On the chip only a kernel input gets its
// padded lanes filled (by the host, from the boundary tile's ttatlas.pad tag);
// pass 7 erases every in-kernel tensor.pad as a no-op. So a reduction that
// reads padded lanes can trust them only when they still hold what the host
// put there. Two questions, answered here for the whole pass:
//  - hostFill (forward): which value should the host put in a load's padded
//    lanes? The identity of the reductions the load reaches unchanged.
//  - maskPaddedReductions (backward): which reduction inputs are not
//    guaranteed to hold the identity in their padded lanes? Those get a mask
//    apply, an elementwise op with a compile-time constant mask, so the padded
//    lanes are right whatever they hold.
#include "LegalizeShapes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace mlir::triton {

namespace {

using ttatlas::PadNeutral;

// A one-input, all-parallel linalg.generic whose body only converts its input
// (arith.extf, arith.truncf, tt.fp_to_fp): what pass 3 makes of a dtype
// conversion. Returns its input operand, or null.
Value conversionInput(Operation *op) {
  if (isa<arith::ExtFOp, arith::TruncFOp, FpToFpOp>(op))
    return op->getOperand(0);
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType type) {
                      return type == utils::IteratorType::parallel;
                    }) ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return {};
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2)
    return {};
  Operation &convert = body.front();
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!isa<arith::ExtFOp, arith::TruncFOp, FpToFpOp>(convert) || !yield ||
      convert.getOperand(0) != body.getArgument(0) ||
      yield.getOperand(0) != convert.getResult(0))
    return {};
  return generic.getDpsInputOperand(0)->get();
}

// The value that carries `use`'s lanes on unchanged (a conversion keeps a
// padded lane's identity, a slice or a pad keeps the lanes it covers), or null
// when the user may change them.
Value passedThrough(OpOperand &use) {
  Operation *op = use.getOwner();
  if (conversionInput(op) == use.get())
    return op->getResult(0);
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
      slice && slice.getSource() == use.get())
    return slice.getResult();
  if (auto pad = dyn_cast<tensor::PadOp>(op);
      pad && pad.getSource() == use.get())
    return pad.getResult();
  return {};
}

bool isReductionLoop(linalg::LinalgOp op, AffineExpr expr) {
  auto dim = dyn_cast<AffineDimExpr>(expr);
  return dim && op.getIteratorTypesArray()[dim.getPosition()] ==
                    utils::IteratorType::reduction;
}

// Does the reduction `op` read dimension `d` of its operand along a reduced
// loop?
bool readsAlongReduction(linalg::LinalgOp op, OpOperand &operand, unsigned d) {
  AffineMap map = op.getMatchingIndexingMap(&operand);
  return d < map.getNumResults() && isReductionLoop(op, map.getResult(d));
}

bool hasReduction(linalg::LinalgOp op) {
  return llvm::is_contained(op.getIteratorTypesArray(),
                            utils::IteratorType::reduction);
}

void collectNeeds(Value value, ArrayRef<bool> padded,
                  SmallVectorImpl<PadNeutral> &needs, unsigned depth) {
  if (depth > 16)
    return;
  for (OpOperand &use : value.getUses()) {
    if (Value next = passedThrough(use)) {
      collectNeeds(next, padded, needs, depth + 1);
      continue;
    }
    auto reader = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!reader || !hasReduction(reader) || !reader.isDpsInput(&use))
      continue;
    bool reads = false;
    for (auto [d, isPadded] : llvm::enumerate(padded))
      reads |= isPadded && readsAlongReduction(reader, use, d);
    if (!reads)
      continue;
    if (std::optional<PadNeutral> identity = operationNeutral(reader))
      if (!llvm::is_contained(needs, *identity))
        needs.push_back(*identity);
  }
}

// What a value's padded lanes are known to hold, in its own coordinates: lane
// (i, j) is real when i < valid[0] and j < valid[1]; the others are padded.
// `holds` is what the host put in them, when the value is a boundary load (or
// a conversion of one, or the tagged pad of one) and nothing changed them;
// unknown otherwise. `unsupported` says why the lanes could not be followed
// (padded lanes reach an op this analysis does not model).
struct Lanes {
  SmallVector<int64_t> valid;
  std::optional<PadNeutral> holds;
  std::string unsupported;
};

bool hasPaddedLanes(const Lanes &lanes, RankedTensorType type) {
  return !llvm::equal(lanes.valid, type.getShape());
}

Lanes lanesOf(Value value, const MaskBoxes &boxes, unsigned depth = 0) {
  auto type = cast<RankedTensorType>(value.getType());
  Lanes lanes;
  lanes.valid.assign(type.getShape().begin(), type.getShape().end());
  Operation *definition = value.getDefiningOp();
  // A tensor argument of a function is an input the host provides, like a
  // load (a kernel's own arguments are pointers; this is for tests of the
  // pass on bare functions).
  if (!definition)
    return lanes;
  if (depth > 24) {
    lanes.unsupported = "the chain of ops that carries its padded lanes is "
                        "too long to follow";
    return lanes;
  }

  if (isa<LoadOp>(definition)) {
    if (auto box = boxes.find(definition); box != boxes.end()) {
      lanes.valid.assign(box->second.bounds.begin(), box->second.bounds.end());
      lanes.holds = box->second.neutral;
    }
    return lanes;
  }
  if (Value input = conversionInput(definition))
    return lanesOf(input, boxes, depth + 1);

  if (auto pad = dyn_cast<tensor::PadOp>(definition)) {
    Lanes source = lanesOf(pad.getSource(), boxes, depth + 1);
    lanes.valid = source.valid;
    lanes.unsupported = source.unsupported;
    // The host fills only what it is given: a pad of the boundary load itself
    // (or of a function argument) is tagged with the fill, and that is what
    // its lanes hold. A pad of anything computed is erased by pass 7, so its
    // new lanes hold whatever the hardware has there.
    Operation *sourceOp = pad.getSource().getDefiningOp();
    if (!sourceOp || isa<LoadOp>(sourceOp)) {
      if (auto tag = pad->getAttrOfType<ttatlas::PadAttr>(ttatlas::kPad))
        lanes.holds = tag.getNeutral();
      else
        lanes.holds = source.holds;
    }
    return lanes;
  }

  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(definition)) {
    Lanes source = lanesOf(slice.getSource(), boxes, depth + 1);
    lanes.unsupported = source.unsupported;
    bool sourcePadded = hasPaddedLanes(source, slice.getSourceType());
    if (llvm::any_of(slice.getStaticOffsets(), ShapedType::isDynamic) ||
        !llvm::all_of(slice.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }) ||
        slice.getType().getRank() != slice.getSourceType().getRank()) {
      if (sourcePadded && lanes.unsupported.empty())
        lanes.unsupported = "a slice with a dynamic offset, a stride or fewer "
                            "dimensions cuts a value with padded lanes";
      return lanes;
    }
    for (auto [d, offset] : llvm::enumerate(slice.getStaticOffsets()))
      lanes.valid[d] = std::clamp<int64_t>(source.valid[d] - offset, 0,
                                           slice.getStaticSizes()[d]);
    lanes.holds = source.holds;
    return lanes;
  }

  // A linalg op: a result dimension is real where every input dimension that
  // the indexing maps tie to the same loop is real. A reduced loop is not in
  // the result; its padded lanes are the mask apply's business, not the
  // result's.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(definition)) {
    SmallVector<Lanes> inputs;
    bool padded = false;
    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      auto inputType = dyn_cast<RankedTensorType>(input->get().getType());
      if (!inputType) {
        inputs.emplace_back();
        continue;
      }
      inputs.push_back(lanesOf(input->get(), boxes, depth + 1));
      if (lanes.unsupported.empty())
        lanes.unsupported = inputs.back().unsupported;
      padded |= hasPaddedLanes(inputs.back(), inputType);
    }
    OpOperand *init =
        linalgOp.getDpsInitOperand(cast<OpResult>(value).getResultNumber());
    AffineMap resultMap = linalgOp.getMatchingIndexingMap(init);
    for (auto [r, expr] : llvm::enumerate(resultMap.getResults())) {
      auto loop = dyn_cast<AffineDimExpr>(expr);
      if (!loop) {
        if (padded && lanes.unsupported.empty())
          lanes.unsupported = "a result index of a linalg op that reads a "
                              "value with padded lanes is not a plain loop";
        continue;
      }
      for (auto [operand, input] :
           llvm::zip_equal(linalgOp.getDpsInputOperands(), inputs)) {
        if (input.valid.empty())
          continue;
        AffineMap map = linalgOp.getMatchingIndexingMap(operand);
        for (auto [k, inputExpr] : llvm::enumerate(map.getResults()))
          if (inputExpr == expr)
            lanes.valid[r] = std::min(lanes.valid[r], input.valid[k]);
      }
    }
    return lanes;
  }

  // Any other op: fine while nothing it reads has padded lanes.
  for (Value operand : definition->getOperands()) {
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType)
      continue;
    Lanes input = lanesOf(operand, boxes, depth + 1);
    if (!input.unsupported.empty())
      lanes.unsupported = input.unsupported;
    else if (hasPaddedLanes(input, operandType))
      lanes.unsupported = "its padded lanes reach an op that this pass cannot "
                          "follow lane by lane (" +
                          definition->getName().getStringRef().str() + ")";
    if (!lanes.unsupported.empty())
      break;
  }
  return lanes;
}

bool isFp8(Type type) {
  auto floating = dyn_cast<FloatType>(getElementTypeOrSelf(type));
  return floating && floating.getWidth() == 8;
}

bool isVpuFloat(Type type) {
  auto floating = dyn_cast<FloatType>(getElementTypeOrSelf(type));
  return floating && (floating.isBF16() || floating.isF32());
}

// The mask apply: `x * M` (identity 0: sum, matmul K) or `x + M` (-inf or +inf:
// max, min), with M a compile-time constant of x's shape that is neutral on the
// real lanes and holds the identity-making value on the padded ones. Exact only
// while the padded lanes of x are finite: 0 * inf is NaN, and inf + -inf too.
// A lane is padded when it is cut off along a dimension in `maskedDims`.
Operation *applyMask(RewriterBase &rewriter, Location loc, Value x,
                     ArrayRef<int64_t> valid, ArrayRef<bool> maskedDims,
                     PadNeutral identity, Value &result) {
  auto type = cast<RankedTensorType>(x.getType());
  auto element = cast<FloatType>(type.getElementType());
  bool multiply = identity == PadNeutral::zero;
  const llvm::fltSemantics &semantics = element.getFloatSemantics();
  llvm::APFloat real = llvm::APFloat::getZero(semantics);
  if (multiply)
    real = llvm::APFloat(semantics, 1);
  llvm::APFloat padded =
      multiply
          ? llvm::APFloat::getZero(semantics)
          : llvm::APFloat::getInf(semantics,
                                  /*Negative=*/identity == PadNeutral::neg_inf);
  SmallVector<Attribute> values;
  ArrayRef<int64_t> shape = type.getShape();
  int64_t count = type.getNumElements();
  for (int64_t flat = 0; flat < count; ++flat) {
    bool cut = false;
    int64_t rest = flat;
    for (int d = shape.size() - 1; d >= 0; --d) {
      int64_t at = rest % shape[d];
      rest /= shape[d];
      cut |= maskedDims[d] && at >= valid[d];
    }
    values.push_back(FloatAttr::get(element, cut ? padded : real));
  }
  auto constant = arith::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(type, values));
  Value empty = tensor::EmptyOp::create(rewriter, loc, shape, element);
  Operation *apply =
      multiply ? linalg::MulOp::create(rewriter, loc,
                                       ValueRange{x, constant.getResult()},
                                       ValueRange{empty})
                     .getOperation()
               : linalg::AddOp::create(rewriter, loc,
                                       ValueRange{x, constant.getResult()},
                                       ValueRange{empty})
                     .getOperation();
  // Mark it, so later passes tell a mask apply from a kernel's own multiply
  // or add by a constant: its constant operand is the mask tile.
  apply->setAttr(ttatlas::kMaskApply, rewriter.getUnitAttr());
  result = apply->getResult(0);
  return apply;
}

} // namespace

std::optional<PadNeutral> hostFill(Value loaded, ArrayRef<int64_t> bounds) {
  auto type = cast<RankedTensorType>(loaded.getType());
  SmallVector<bool> padded;
  for (auto [bound, size] : llvm::zip_equal(bounds, type.getShape()))
    padded.push_back(bound < size);
  SmallVector<PadNeutral> needs;
  collectNeeds(loaded, padded, needs, 0);
  if (needs.empty())
    return std::nullopt;
  return needs.front();
}

LogicalResult maskPaddedReductions(RewriterBase &rewriter, linalg::LinalgOp op,
                                   const MaskBoxes &boxes,
                                   SmallVectorImpl<Operation *> &created) {
  if (!hasReduction(op))
    return success();
  for (OpOperand *operand : op.getDpsInputOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand->get().getType());
    if (!type)
      continue;
    Lanes lanes = lanesOf(operand->get(), boxes);
    bool readAlongReduction = false;
    for (unsigned d = 0; d < type.getRank(); ++d)
      readAlongReduction |= readsAlongReduction(op, *operand, d);
    if (readAlongReduction && !lanes.unsupported.empty()) {
      ttatlas::emitUnsupported(
          op, "cannot tell which lanes of a reduction input are padded: " +
                  lanes.unsupported);
      return failure();
    }
    SmallVector<bool> maskedDims;
    for (auto [d, size] : llvm::enumerate(type.getShape()))
      maskedDims.push_back(lanes.valid[d] < size &&
                           readsAlongReduction(op, *operand, d));
    if (!llvm::is_contained(maskedDims, true))
      continue;
    std::string why;
    std::optional<PadNeutral> identity = operationNeutral(op, {}, &why);
    if (!identity) {
      ttatlas::emitUnsupported(
          op, "cannot mask padded lanes before the reduction: " + why);
      return failure();
    }
    // Guaranteed: the host put the identity there and nothing changed it.
    if (lanes.holds == identity)
      continue;
    if (*identity == PadNeutral::one) {
      ttatlas::emitUnsupported(
          op, "padded lanes before a product reduction cannot be masked: the "
              "VPU has no op that writes a 1 into chosen lanes");
      return failure();
    }

    // The VPU computes in bf16 and f32. An fp8 value is masked before its
    // conversion, when there is one.
    Value target = operand->get();
    Operation *converter = nullptr;
    if (isFp8(target.getType())) {
      converter = target.getDefiningOp();
      Value input = converter ? conversionInput(converter) : Value();
      if (!input || !isVpuFloat(input.getType())) {
        ttatlas::emitUnsupported(
            op, "padded lanes of an fp8 reduction input cannot be masked: "
                "the VPU cannot multiply fp8 and the value is not converted "
                "from bf16 or f32 in the kernel");
        return failure();
      }
      target = input;
    } else if (!isVpuFloat(target.getType())) {
      ttatlas::emitUnsupported(
          op, "padded lanes of a reduction input cannot be masked: the VPU "
              "computes in bf16 and f32 only");
      return failure();
    }

    rewriter.setInsertionPoint(op);
    Value masked;
    created.push_back(applyMask(rewriter, op.getLoc(), target, lanes.valid,
                                maskedDims, *identity, masked));
    Value replacement = masked;
    if (converter) {
      IRMapping mapping;
      mapping.map(target, masked);
      replacement = rewriter.clone(*converter, mapping)->getResult(0);
    }
    rewriter.modifyOpInPlace(op, [&] { operand->set(replacement); });
    ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                "mask padded lanes before a reduction");
  }
  return success();
}

} // namespace mlir::triton
