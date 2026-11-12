// Pass 4: tile linalg at the Atlas tile size, fuse sliced tensor producers,
// fully unroll the static tile loops, and attach the MXU policy tag.
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
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

static std::optional<int64_t> constantInteger(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "tt.splat" || name == "tt.broadcast") &&
      definition->getNumOperands() == 1)
    return constantInteger(definition->getOperand(0));
  if (name != "arith.constant")
    return std::nullopt;
  auto integer = dyn_cast_or_null<IntegerAttr>(definition->getAttr("value"));
  if (integer)
    return integer.getInt();
  return std::nullopt;
}

static std::optional<int64_t> rangeAxis(Value value, unsigned rank) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.make_range")
    return rank >= 1 ? std::optional<int64_t>(0) : std::nullopt;
  if (name == "tt.expand_dims" && definition->getNumOperands() == 1) {
    auto axis = dyn_cast_or_null<IntegerAttr>(definition->getAttr("axis"));
    auto prior = rangeAxis(definition->getOperand(0), rank - 1);
    if (!axis || !prior)
      return std::nullopt;
    return *prior >= axis.getInt() ? *prior + 1 : *prior;
  }
  if (name == "tt.broadcast" && definition->getNumOperands() == 1) {
    auto sourceType = dyn_cast<RankedTensorType>(definition->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(value.getType());
    if (!sourceType || !resultType || sourceType.getRank() > resultType.getRank())
      return std::nullopt;
    auto prior = rangeAxis(definition->getOperand(0), sourceType.getRank());
    if (!prior)
      return std::nullopt;
    return *prior + resultType.getRank() - sourceType.getRank();
  }
  return std::nullopt;
}

static std::optional<int64_t> rangeBound(Value value, int64_t extent) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "tt.make_range")
    return std::nullopt;
  auto start = dyn_cast_or_null<IntegerAttr>(definition->getAttr("start"));
  auto end = dyn_cast_or_null<IntegerAttr>(definition->getAttr("end"));
  if (!start || !end || start.getInt() != 0)
    return std::nullopt;
  return std::min(end.getInt(), extent);
}

static bool collectMaskBounds(Value mask, ArrayRef<int64_t> fullShape,
                              SmallVectorImpl<int64_t> &logicalShape) {
  if (auto andOp = mask.getDefiningOp()) {
    StringRef name = andOp->getName().getStringRef();
    if ((name == "tt.broadcast" || name == "tt.expand_dims") &&
        andOp->getNumOperands() == 1)
      return collectMaskBounds(andOp->getOperand(0), fullShape, logicalShape);
    if (andOp->getName().getStringRef() == "arith.andi" &&
        andOp->getNumOperands() == 2) {
      SmallVector<int64_t> lhs(fullShape), rhs(fullShape);
      if (!collectMaskBounds(andOp->getOperand(0), fullShape, lhs) ||
          !collectMaskBounds(andOp->getOperand(1), fullShape, rhs))
        return false;
      for (auto [index, extent] : llvm::enumerate(fullShape))
        logicalShape[index] = std::min(lhs[index], rhs[index]);
      return true;
    }
    if (andOp->getName().getStringRef() == "arith.cmpi" &&
        andOp->getNumOperands() == 2) {
      auto predicate = dyn_cast<arith::CmpIOp>(andOp);
      if (!predicate || predicate.getPredicate() != arith::CmpIPredicate::slt)
        return false;
      Value range = andOp->getOperand(0);
      auto axis = rangeAxis(range, fullShape.size());
      auto bound = constantInteger(andOp->getOperand(1));
      if (!axis || !bound || *axis < 0 || *axis >= fullShape.size())
        return false;
      auto start = rangeBound(range, fullShape[*axis]);
      if (!start)
        return false;
      int64_t logical = std::max<int64_t>(0, std::min(*bound, *start));
      logicalShape[*axis] = std::min(logicalShape[*axis], logical);
      return true;
    }
  }
  Operation *definition = mask.getDefiningOp();
  if (definition && definition->getName().getStringRef() == "arith.constant") {
    auto dense = dyn_cast_or_null<DenseElementsAttr>(definition->getAttr("value"));
    return dense && dense.isSplat() &&
           dense.getSplatValue<llvm::APInt>().isAllOnes();
  }
  return false;
}

static std::optional<ttatlas::PadNeutral> constantNeutral(Value value) {
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
    return integer.getInt() == 0
               ? std::optional(ttatlas::PadNeutral::zero)
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

static std::optional<ttatlas::PadNeutral> operationNeutral(Operation *op) {
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
      inferred = reduction ? ttatlas::PadNeutral::neg_inf
                           : ttatlas::PadNeutral::zero;
  });
  return inferred;
}

static LogicalResult padToAtlasTiles(IRRewriter &rewriter,
                                     linalg::LinalgOp op,
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
        ttatlas::emitUnsupported(
            op, "negative-infinity padding requires floats");
        return failure();
      }
      value = FloatAttr::get(
          floatType,
          llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                /*Negative=*/true));
    } else {
      value = rewriter.getZeroAttr(elementType);
    }
    paddingValues.push_back(value);
  }

  // Triton 3.9's pinned MLIR models linalg padding through TilingInterface.
  // Every iterator dimension is rounded up to the Atlas tile size.
  linalg::PadTilingInterfaceOptions options;
  options.setPaddingSizes(paddingSizes);
  options.setPadToMultipleOf(true);
  options.setPaddingValues(paddingValues);
  FailureOr<linalg::PadTilingInterfaceResult> padded =
      linalg::rewriteAsPaddedOp(rewriter, op, options);
  if (failed(padded)) {
    ttatlas::emitUnsupported(
        op, "MLIR failed to pad linalg iteration dimensions");
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

static std::optional<ttatlas::PadNeutral> requiredNeutral(Value value,
                                                          unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "arith.truncf" || name == "arith.extf" ||
       name == "tt.fp_to_fp") && definition->getNumOperands() == 1)
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

static LogicalResult tagMaskedBoundaries(
    ModuleOp module, DenseMap<Operation *, Attribute> &storePadTags) {
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
    SmallVector<int64_t> logicalShape(tensor.getShape().begin(),
                                      tensor.getShape().end());
    if (!collectMaskBounds(mask, tensor.getShape(), logicalShape)) {
      ttatlas::emitUnsupported(
          op, "edge mask is not a static make_range comparison against a constant");
      failed = true;
      return;
    }
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
      std::optional<ttatlas::PadNeutral> consumerNeutral = requiredNeutral(value);
      if (!consumerNeutral || *consumerNeutral != *neutral) {
        ttatlas::emitUnsupported(
            op, "masked load neutral element does not match its consumer");
        failed = true;
        return;
      }
    }
    Operation *taggedOp = isLoad ? op : value.getDefiningOp();
    if (!taggedOp) {
      ttatlas::emitUnsupported(op, "padded store value has no defining operation");
      failed = true;
      return;
    }
    auto pad = ttatlas::PadAttr::get(op->getContext(), *neutral, logicalShape);
    if (Attribute old = taggedOp->getAttr(ttatlas::kPad); old && old != pad) {
      ttatlas::emitUnsupported(taggedOp,
                               "conflicting ttatlas.pad metadata from edge masks");
      failed = true;
      return;
    }
    auto [it, inserted] = tags.try_emplace(taggedOp, pad);
    if (!inserted && it->second != pad) {
      ttatlas::emitUnsupported(taggedOp,
                               "conflicting ttatlas.pad metadata from edge masks");
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

static LogicalResult propagateStorePadTags(
    const DenseMap<Operation *, Attribute> &storePadTags) {
  for (auto [store, pad] : storePadTags) {
    Value storedValue = store->getOperand(1);
    Operation *definition = storedValue.getDefiningOp();
    if (!definition) {
      ttatlas::emitUnsupported(
          store, "padded store value has no defining operation after tiling");
      return failure();
    }
    if (Attribute old = definition->getAttr(ttatlas::kPad); old && old != pad) {
      ttatlas::emitUnsupported(
          definition, "conflicting ttatlas.pad metadata after tiling");
      return failure();
    }
    definition->setAttr(ttatlas::kPad, pad);
  }
  return success();
}

static bool setMxuZero(Operation *op) {
  Attribute oldUnit = op->getAttr(ttatlas::kMxu);
  auto oldInteger = dyn_cast_or_null<IntegerAttr>(oldUnit);
  if (oldUnit && (!oldInteger || !oldInteger.getType().isSignlessInteger(32) ||
                  oldInteger.getInt() != 0)) {
    ttatlas::emitUnsupported(
        op, "linalg.matmul has conflicting ttatlas.mxu; expected 0");
    return false;
  }
  // atlas-hw: MXU0 is the pass 4 default policy; MXU1 cannot run a K chain.
  op->setAttr(ttatlas::kMxu,
              IntegerAttr::get(IntegerType::get(op->getContext(), 32), 0));
  return true;
}

static LogicalResult fuseSlicedProducers(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> consumers;
    module.walk([&](Operation *op) {
      if (isLinalgCompute(op))
        consumers.push_back(op);
    });
    for (Operation *consumer : consumers) {
      for (OpOperand &operand : consumer->getOpOperands()) {
        auto slice = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
        if (!slice)
          continue;
        auto producer = slice.getSource().getDefiningOp<linalg::LinalgOp>();
        if (!producer || producer->getNumResults() != 1)
          continue;
        rewriter.setInsertionPoint(consumer);
        if (succeeded(linalg::fuseProducerOfTensor(
                rewriter, producer->getResult(0), operand))) {
          changed = true;
          break;
        }
      }
      if (changed)
        break;
    }
  }
  return success();
}

static LogicalResult unrollTileLoops(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  // Inner loops first, so a matmul's K accumulator chain is explicit before
  // its enclosing M/N tiles are replicated.
  for (scf::ForOp loop : llvm::reverse(loops))
    if (failed(loopUnrollFull(loop))) {
      ttatlas::emitUnsupported(
          loop, "tile loop does not have a statically known trip count");
      return failure();
    }
  return success();
}

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
          ttatlas::emitUnsupported(
              op, "linalg tensor must have static rank 1 or 2 and positive dimensions");
          hadFailure = true;
          validTensorShapes = false;
          break;
        }
      }
      if (!validTensorShapes)
        continue;
      Attribute oldUnit = op->getAttr(ttatlas::kMxu);
      auto oldInteger = dyn_cast_or_null<IntegerAttr>(oldUnit);
      if (op->getName().getStringRef() == "linalg.matmul" && oldUnit &&
          (!oldInteger || !oldInteger.getType().isSignlessInteger(32) ||
           oldInteger.getInt() != 0)) {
        ttatlas::emitUnsupported(
            op, "linalg.matmul has conflicting ttatlas.mxu; expected 0");
        hadFailure = true;
      }
    }
    if (hadFailure) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(module.getContext());
    // Tile consumers first. Their constant extract_slice operands are the
    // anchors the upstream tensor-producer fusion utility uses.
    for (Operation *op : llvm::reverse(computeOps)) {
      if (op->getBlock() == nullptr)
        continue;
      if (llvm::all_of(op->getResults(), [](Value result) {
            return result.use_empty();
          })) {
        rewriter.eraseOp(op);
        continue;
      }
      auto linalgOp = cast<linalg::LinalgOp>(op);
      if (op->getName().getStringRef() == "linalg.matmul" && !setMxuZero(op)) {
        signalPassFailure();
        return;
      }
      bool needsPadding = llvm::any_of(linalgOp.getStaticLoopRanges(),
                                       [](int64_t size) {
        return size % ttatlas::target::kTileSize != 0;
      });
      if (needsPadding) {
        linalg::LinalgOp paddedOp;
        SmallVector<Value> replacements;
        rewriter.setInsertionPoint(op);
        if (failed(padToAtlasTiles(rewriter, linalgOp, paddedOp,
                                   replacements))) {
          signalPassFailure();
          return;
        }
        rewriter.replaceOp(op, replacements);
        op = paddedOp.getOperation();
        linalgOp = paddedOp;
      }
      SmallVector<int64_t> loopSizes(linalgOp.getNumLoops(),
                                     ttatlas::target::kTileSize);
      bool needsTiling = llvm::any_of(linalgOp.getStaticLoopRanges(),
                                      [](int64_t size) {
                                        return size > ttatlas::target::kTileSize;
                                      });
      if (!needsTiling)
        continue;

      linalg::LinalgTilingOptions options;
      options.setTileSizes(loopSizes);
      options.setLoopType(linalg::LinalgTilingLoopType::Loops);
      rewriter.setInsertionPoint(op);
      auto tiled = linalg::tileLinalgOp(rewriter, linalgOp, options);
      if (::mlir::failed(tiled)) {
        ttatlas::emitUnsupported(op, "MLIR failed to tile the linalg operation");
        hadFailure = true;
        continue;
      }
      if (tiled->tensorResults.size() != op->getNumResults()) {
        ttatlas::emitUnsupported(op, "MLIR tiling returned an unexpected result count");
        hadFailure = true;
        continue;
      }
      ttatlas::recordLegalization(op, "ttatlas-legalize-shapes",
                                   "tile linalg compute to 32x32");
      rewriter.replaceOp(op, tiled->tensorResults);
      if (failed(fuseSlicedProducers(module))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(fuseSlicedProducers(module)) || failed(unrollTileLoops(module))) {
      signalPassFailure();
      return;
    }

    if (failed(propagateStorePadTags(storePadTags))) {
      signalPassFailure();
      return;
    }

    module.walk([&](Operation *op) {
      if (isLinalgCompute(op) && op->getName().getStringRef() == "linalg.matmul")
        hadFailure |= !setMxuZero(op);
    });
    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
