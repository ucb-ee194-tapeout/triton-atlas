// Pass 7 lowering for pointwise linalg.generic operations over BF16 tiles.
#include "Patterns.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir::triton {
namespace {

static Operation *createAtlasOp(ConversionPatternRewriter &rewriter,
                                Location location, StringRef name,
                                ValueRange operands, Type resultType,
                                NamedAttribute attribute) {
  OperationState state(location, name);
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttribute(attribute.getName(), attribute.getValue());
  return rewriter.create(state);
}

static bool isZeroConstant(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "arith.constant")
    return false;
  Attribute attribute = definition->getAttr("value");
  if (auto floating = dyn_cast_or_null<FloatAttr>(attribute))
    return floating.getValue().isZero();
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(attribute))
    return dense.isSplat() && dense.getSplatValue<llvm::APFloat>().isZero();
  return false;
}

static StringRef binaryKind(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "arith.addf")
    return "add";
  if (name == "arith.subf")
    return "sub";
  if (name == "arith.mulf")
    return "mul";
  if (name == "arith.minnumf")
    return "min";
  if (name == "arith.maxnumf")
    return "max";
  return {};
}

static SmallVector<Operation *> collectInputConstants(Operation *op) {
  SmallVector<Operation *> constants;
  for (Value operand : op->getOperands())
    if (Operation *definition = operand.getDefiningOp();
        definition && definition->getName().getStringRef() == "arith.constant")
      constants.push_back(definition);
  llvm::sort(constants);
  constants.erase(std::unique(constants.begin(), constants.end()),
                  constants.end());
  return constants;
}

static void eraseDeadConstants(ArrayRef<Operation *> constants,
                               ConversionPatternRewriter &rewriter) {
  for (Operation *constant : constants)
    if (llvm::all_of(constant->getResults(),
                     [](Value result) { return result.use_empty(); }))
      rewriter.eraseOp(constant);
}

static Operation *genericDestinationToErase(Operation *generic,
                                            Value destination) {
  Operation *empty = destination.getDefiningOp();
  if (!empty || empty->getName().getStringRef() != "tensor.empty" ||
      !empty->getResult(0).hasOneUse() ||
      *empty->getResult(0).getUsers().begin() != generic)
    return nullptr;
  // The generic has just been replaced, but can remain visible to the
  // rewriter until this transaction commits. Its fresh tensor.empty is
  // nevertheless dead in the replacement program.
  return empty;
}

struct EraseTileCast : ConversionPattern {
  EraseTileCast(const TypeConverter &converter, MLIRContext *context,
                StringRef rootName)
      : ConversionPattern(converter, rootName, PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || operands.size() != 1 ||
        !isa<atlas::VirtualBF16Type>(operands.front().getType()) ||
        !isa<atlas::VirtualBF16Type>(
            getTypeConverter()->convertType(op->getResult(0).getType())))
      return failure();
    rewriter.replaceOp(op, operands.front());
    return success();
  }
};

struct EraseDeadTensorOp : ConversionPattern {
  EraseDeadTensorOp(const TypeConverter &converter, MLIRContext *context,
                    StringRef rootName)
      : ConversionPattern(converter, rootName, PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::any_of(op->getResults(),
                     [](Value result) { return !result.use_empty(); }))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// A value that already is a physically broadcast tile: a row reduction's result
// (virtual_vpu_reduce fills every row with it) or a 1x32 bias row (the host
// repeats it over the rows), found through the ops that keep the tile.
static bool isBroadcastSource(Value value, unsigned depth = 0) {
  Operation *definition = value.getDefiningOp();
  if (!definition || depth > 8)
    return false;
  if (auto slot = definition->getAttrOfType<ttatlas::SlotAttr>(ttatlas::kSlot))
    return slot.getRole() == ttatlas::SlotRole::bias_bf16;
  if (definition->getName().getStringRef() == "linalg.reduce")
    return true;
  if (isa<arith::TruncFOp, arith::ExtFOp>(definition) &&
      definition->getNumOperands() == 1)
    return isBroadcastSource(definition->getOperand(0), depth + 1);
  return false;
}

struct EraseTileBroadcast : ConversionPattern {
  EraseTileBroadcast(const TypeConverter &converter, MLIRContext *context,
                     StringRef rootName)
      : ConversionPattern(converter, rootName, PatternBenefit(2), context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || operands.empty() ||
        !isa<atlas::VirtualBF16Type>(operands.front().getType()) ||
        !isa<atlas::VirtualBF16Type>(
            getTypeConverter()->convertType(op->getResult(0).getType())))
      return failure();
    // The tile is only repeated along the broadcast dimension in the hardware
    // tile when its source was already laid out that way.
    if (!isBroadcastSource(op->getOperand(0)))
      return ttatlas::emitUnsupported(
          op, "a broadcast is lowered only for a row reduction's result or a "
              "bias row, which are already laid out across the tile");
    rewriter.replaceOp(op, operands.front());
    return success();
  }
};

// An op with no Atlas lowering yet: say so, with its name, rather than leave
// the whole kernel 'failed to legalize'.
struct RejectUnlowered : ConversionPattern {
  RejectUnlowered(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), PatternBenefit(0),
                          context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &) const override {
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "linalg" && dialect != "tt" && dialect != "scf" &&
        dialect != "math" && dialect != "tensor")
      return failure();
    // Ops with their own patterns report their own reasons.
    if (isa<FuncOp, LoadOp, StoreOp, linalg::GenericOp, linalg::ReduceOp,
            linalg::MatmulOp, linalg::BroadcastOp, linalg::FillOp,
            linalg::YieldOp, tensor::EmptyOp, tensor::PadOp,
            tensor::ExtractSliceOp, tensor::InsertSliceOp,
            tensor::ExpandShapeOp>(op))
      return failure();
    if (op->hasAttr(ttatlas::kMaskApply))
      return ttatlas::emitUnsupported(
          op, "a ttatlas.mask_apply needs its mask tile as a slot, which "
              "pass 6 does not assign yet");
    return ttatlas::emitUnsupported(op,
                                    Twine("pass 7 has no Atlas lowering for ") +
                                        op->getName().getStringRef() + " yet");
  }
};

struct ConvertGeneric : ConversionPattern {
  ConvertGeneric(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.generic", PatternBenefit(1),
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
        op->getNumResults() != 1 || op->getNumOperands() < 2)
      return failure();

    auto generic = cast<linalg::GenericOp>(op);
    if (!llvm::all_of(generic.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }) ||
        !llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return type == utils::IteratorType::parallel;
                      }))
      return ttatlas::emitUnsupported(
          op, "pass 7 lowers only pointwise generics: every indexing map "
              "must be the identity and every loop parallel (a broadcast or "
              "transposed operand has no lowering yet)");
    Block &body = op->getRegion(0).front();
    Operation *compute = nullptr;
    for (Operation &nested : body) {
      if (nested.getName().getStringRef() == "linalg.yield" ||
          nested.getName().getStringRef() == "arith.constant")
        continue;
      if (compute)
        return ttatlas::emitUnsupported(
            op, "a linalg.generic with several operations in its body has no "
                "Atlas lowering yet; pass 7 lowers one pointwise operation "
                "per generic");
      compute = &nested;
    }
    if (!compute || compute->getNumResults() != 1)
      return failure();

    unsigned inputCount = op->getNumOperands() - op->getNumResults();

    // A dtype conversion between tile values (pass 5's readout marker, a
    // widening or narrowing cast): every tile is BF16 on the VPU, so it keeps
    // the value it reads.
    if (isa<arith::TruncFOp, arith::ExtFOp>(compute)) {
      auto input = dyn_cast<BlockArgument>(compute->getOperand(0));
      if (inputCount != 1 || !input || input.getArgNumber() != 0 ||
          !isa<atlas::VirtualBF16Type>(operands.front().getType()))
        return failure();
      Operation *empty = genericDestinationToErase(
          op, op->getOperand(op->getNumOperands() - 1));
      Type convertedResult =
          getTypeConverter()->convertType(op->getResult(0).getType());
      // A BF16 -> FP8 conversion is the boundary between consecutive MLP
      // layers.  Atlas supplies pack_fp8 specifically for this transition;
      // the conversion's scale tag was attached by pass 3.
      if (isa<atlas::VirtualFP8Type>(convertedResult)) {
        auto scale = compute->getAttrOfType<IntegerAttr>(ttatlas::kScale);
        if (!scale)
          return ttatlas::emitUnsupported(
              op, "BF16 to FP8 conversion is missing ttatlas.scale");
        Operation *pack = createAtlasOp(
            rewriter, op->getLoc(), atlas::VirtualPackFP8Op::getOperationName(),
            ValueRange{operands.front()}, convertedResult,
            rewriter.getNamedAttr("scale_code",
                                 rewriter.getI32IntegerAttr(scale.getInt())));
        rewriter.replaceOp(op, pack->getResults());
        if (empty)
          rewriter.eraseOp(empty);
        return success();
      }
      if (!isa<atlas::VirtualBF16Type>(convertedResult))
        return failure();
      rewriter.replaceOp(op, operands.front());
      if (empty)
        rewriter.eraseOp(empty);
      return success();
    }
    if (compute->getNumOperands() != 2)
      return failure();
    SmallVector<Value> inputs;
    for (unsigned index = 0; index < inputCount; ++index)
      inputs.push_back(operands[index]);

    Type outputType = atlas::VirtualBF16Type::get(op->getContext());
    if (compute->getName().getStringRef() == "arith.maxnumf") {
      for (unsigned zeroSide = 0; zeroSide != 2; ++zeroSide) {
        Value zero = compute->getOperand(zeroSide);
        bool isZero = isZeroConstant(zero);
        if (auto blockArgument = dyn_cast<BlockArgument>(zero);
            blockArgument && blockArgument.getArgNumber() < inputCount)
          isZero |=
              isZeroConstant(op->getOperand(blockArgument.getArgNumber()));
        if (!isZero)
          continue;
        auto tensorArgument =
            dyn_cast<BlockArgument>(compute->getOperand(1 - zeroSide));
        if (!tensorArgument || tensorArgument.getArgNumber() >= inputCount)
          continue;
        Value source = inputs[tensorArgument.getArgNumber()];
        Operation *relu = createAtlasOp(
            rewriter, op->getLoc(),
            atlas::VirtualVPUUnaryOp::getOperationName(), ValueRange{source},
            outputType,
            rewriter.getNamedAttr("kind", rewriter.getStringAttr("relu")));
        Value destination = op->getOperand(op->getNumOperands() - 1);
        Operation *empty = genericDestinationToErase(op, destination);
        SmallVector<Operation *> constants = collectInputConstants(op);
        rewriter.replaceOp(op, relu->getResults());
        eraseDeadConstants(constants, rewriter);
        if (empty)
          rewriter.eraseOp(empty);
        return success();
      }
    }

    StringRef kind = binaryKind(compute);
    if (kind.empty())
      return ttatlas::emitUnsupported(
          op, Twine("no Atlas VPU operation for ") +
                  compute->getName().getStringRef() + " yet");
    auto lhs = dyn_cast<BlockArgument>(compute->getOperand(0));
    auto rhs = dyn_cast<BlockArgument>(compute->getOperand(1));
    if (!lhs || !rhs || lhs.getArgNumber() >= inputCount ||
        rhs.getArgNumber() >= inputCount)
      return failure();

    Operation *binary = createAtlasOp(
        rewriter, op->getLoc(), atlas::VirtualVPUBinaryOp::getOperationName(),
        ValueRange{inputs[lhs.getArgNumber()], inputs[rhs.getArgNumber()]},
        outputType,
        rewriter.getNamedAttr("kind", rewriter.getStringAttr(kind)));
    Value destination = op->getOperand(op->getNumOperands() - 1);
    Operation *empty = genericDestinationToErase(op, destination);
    SmallVector<Operation *> constants = collectInputConstants(op);
    rewriter.replaceOp(op, binary->getResults());
    eraseDeadConstants(constants, rewriter);
    if (empty)
      rewriter.eraseOp(empty);
    return success();
  }
};

} // namespace

void populateElementwisePatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  patterns.add<ConvertGeneric>(typeConverter, patterns.getContext());
  patterns.add<RejectUnlowered>(typeConverter, patterns.getContext());
  patterns.add<EraseTileCast>(typeConverter, patterns.getContext(),
                              "arith.truncf");
  patterns.add<EraseTileCast>(typeConverter, patterns.getContext(),
                              "arith.extf");
  patterns.add<EraseTileBroadcast>(typeConverter, patterns.getContext(),
                                   "linalg.broadcast");
  patterns.add<EraseTileBroadcast>(typeConverter, patterns.getContext(),
                                   "tensor.expand_shape");
  for (StringRef tensorOp : {"tensor.pad", "tensor.extract_slice",
                             "tensor.insert_slice", "tensor.empty"})
    patterns.add<EraseDeadTensorOp>(typeConverter, patterns.getContext(),
                                    tensorOp);
}

} // namespace mlir::triton
