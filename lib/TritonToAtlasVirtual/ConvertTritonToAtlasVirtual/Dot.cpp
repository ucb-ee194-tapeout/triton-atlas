// Pass 7 lowering for MXU matmuls. The ordinary two-tile form is a virtual
// matmul; multi-K forms become an explicitly ordered MXU resource chain.
#include "Patterns.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace mlir::triton {
namespace {

using Role = ttatlas::SlotRole;

static std::optional<Role> getWeightRole(Value value) {
  llvm::SmallPtrSet<Operation *, 8> seen;
  while (Operation *definition = value.getDefiningOp()) {
    if (!seen.insert(definition).second)
      return std::nullopt;
    if (auto slot = definition->getAttrOfType<ttatlas::SlotAttr>(
            ttatlas::kSlot))
      return slot.getRole();
    StringRef name = definition->getName().getStringRef();
    if ((name == "linalg.transpose" || name == "tensor.extract_slice" ||
         name == "tensor.cast" || name == "tensor.pad") &&
        definition->getNumOperands() > 0) {
      value = definition->getOperand(0);
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static FailureOr<int32_t> getMxuUnit(Operation *matmul) {
  auto unit = matmul->getAttrOfType<IntegerAttr>(ttatlas::kMxu);
  if (!unit || !unit.getType().isSignlessInteger(32) || unit.getInt() < 0 ||
      unit.getInt() >= ttatlas::target::kMxuUnits) {
    ttatlas::emitUnsupported(matmul,
                             "linalg.matmul requires a valid ttatlas.mxu i32");
    return failure();
  }
  return static_cast<int32_t>(unit.getInt());
}

static Operation *createAtlasOp(ConversionPatternRewriter &rewriter,
                                Location location, StringRef name,
                                ValueRange operands, TypeRange resultTypes,
                                ArrayRef<NamedAttribute> attributes = {}) {
  OperationState state(location, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(attributes);
  return rewriter.create(state);
}

static bool isZeroFill(Operation *fill) {
  if (!fill || fill->getName().getStringRef() != "linalg.fill" ||
      fill->getNumOperands() < 2)
    return false;
  Operation *constant = fill->getOperand(0).getDefiningOp();
  if (!constant || constant->getName().getStringRef() != "arith.constant")
    return false;
  Attribute value = constant->getAttr("value");
  if (auto floating = dyn_cast_or_null<FloatAttr>(value))
    return floating.getValue().isZero();
  if (auto integer = dyn_cast_or_null<IntegerAttr>(value))
    return integer.getInt() == 0;
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(value)) {
    if (!dense.isSplat())
      return false;
    if (isa<FloatType>(dense.getElementType()))
      return dense.getSplatValue<llvm::APFloat>().isZero();
    if (isa<IntegerType>(dense.getElementType()))
      return dense.getSplatValue<llvm::APInt>().isZero();
  }
  return false;
}

static void eraseZeroFill(Operation *fill,
                          ConversionPatternRewriter &rewriter) {
  if (fill->getNumResults() != 1 || !fill->getResult(0).use_empty())
    return;
  Operation *scalar = fill->getOperand(0).getDefiningOp();
  Value destination = fill->getOperand(1);
  rewriter.eraseOp(fill);
  if (Operation *empty = destination.getDefiningOp();
      empty && empty->getName().getStringRef() == "tensor.empty" &&
      empty->getResult(0).use_empty())
    rewriter.eraseOp(empty);
  if (scalar && scalar->getName().getStringRef() == "arith.constant" &&
      llvm::all_of(scalar->getResults(), [](Value result) {
        return result.use_empty();
      }))
    rewriter.eraseOp(scalar);
}

static LogicalResult replaceMatmul(Operation *matmul,
                                   ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) {
  if (matmul->getNumOperands() != 3 || operands.size() != 3 ||
      matmul->getNumResults() != 1)
    return ttatlas::emitUnsupported(matmul,
                                    "expected a single-result linalg.matmul");

  FailureOr<int32_t> unit = getMxuUnit(matmul);
  if (failed(unit))
    return failure();
  std::optional<Role> role = getWeightRole(matmul->getOperand(1));
  if (!role || (*role != Role::weight_fp8_nk &&
                *role != Role::weight_fp8_kn))
    return ttatlas::emitUnsupported(
        matmul, "matmul weight requires a weight_fp8_nk or weight_fp8_kn slot");
  if (*role == Role::weight_fp8_kn)
    if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
        transpose &&
        transpose->getName().getStringRef() == "linalg.transpose")
      return ttatlas::emitUnsupported(
          matmul, "weight_fp8_kn must not have a linalg.transpose producer");
  auto activationRole = getWeightRole(matmul->getOperand(0));
  if (!activationRole || *activationRole != Role::activation_fp8)
    return ttatlas::emitUnsupported(
        matmul, "matmul left operand requires an activation_fp8 slot");

  Operation *fill = matmul->getOperand(2).getDefiningOp();
  if (!isZeroFill(fill) || !fill->getResult(0).hasOneUse())
    return ttatlas::emitUnsupported(
        matmul, "matmul accumulator must come from an unshared linalg.fill zero");

  Value activation = operands[0];
  Value weight = operands[1];
  if (*role == Role::weight_fp8_nk)
    if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
        transpose &&
        transpose->getName().getStringRef() == "linalg.transpose")
      weight = rewriter.getRemappedValue(transpose->getOperand(0));
  if (*role == Role::weight_fp8_kn) {
    Type fp8 = atlas::VirtualFP8Type::get(matmul->getContext());
    Operation *transpose = createAtlasOp(
        rewriter, matmul->getLoc(), atlas::VirtualXLUTransposeOp::getOperationName(),
        ValueRange{weight}, TypeRange{fp8});
    weight = transpose->getResult(0);
  }
  Type bf16 = atlas::VirtualBF16Type::get(matmul->getContext());
  Builder builder(matmul->getContext());
  Operation *lowered = createAtlasOp(
      rewriter, matmul->getLoc(), atlas::VirtualMXUMatmulOp::getOperationName(),
      ValueRange{activation, weight}, TypeRange{bf16},
      ArrayRef<NamedAttribute>{
          builder.getNamedAttr("unit", builder.getI32IntegerAttr(*unit))});

  rewriter.replaceOp(matmul, lowered->getResults());
  eraseZeroFill(fill, rewriter);
  return success();
}

static SmallVector<Operation *> collectKChain(Operation *root) {
  SmallVector<Operation *> chain{root};
  Operation *current = root;
  while (current->getNumResults() == 1) {
    Value result = current->getResult(0);
    Operation *next = nullptr;
    for (OpOperand &use : result.getUses()) {
      Operation *user = use.getOwner();
      if (user->getName().getStringRef() != "linalg.matmul" ||
          use.getOperandNumber() != 2)
        continue;
      if (next)
        return {};
      next = user;
    }
    if (!next)
      break;
    if (!result.hasOneUse())
      return {};
    chain.push_back(next);
    current = next;
  }
  return chain;
}

static Value latestStateBefore(Operation *anchor) {
  Value state;
  for (Operation &op : *anchor->getBlock()) {
    if (&op == anchor)
      break;
    for (Value result : op.getResults())
      if (isa<atlas::VirtualStateType>(result.getType()))
        state = result;
  }
  return state;
}

static LogicalResult lowerKChain(Operation *root, ArrayRef<Operation *> chain,
                                 ConversionPatternRewriter &rewriter) {
  SmallVector<int32_t> units;
  SmallVector<Role> roles;
  for (Operation *matmul : chain) {
    FailureOr<int32_t> unit = getMxuUnit(matmul);
    if (failed(unit))
      return failure();
    units.push_back(*unit);
    auto role = getWeightRole(matmul->getOperand(1));
    if (!role || (*role != Role::weight_fp8_nk &&
                  *role != Role::weight_fp8_kn))
      return ttatlas::emitUnsupported(
          matmul, "K-chain weight requires a weight_fp8_nk or weight_fp8_kn slot");
    roles.push_back(*role);
    if (*role == Role::weight_fp8_kn)
      if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
          transpose &&
          transpose->getName().getStringRef() == "linalg.transpose")
        return ttatlas::emitUnsupported(
            matmul,
            "weight_fp8_kn must not have a linalg.transpose producer");
    auto activationRole = getWeightRole(matmul->getOperand(0));
    if (!activationRole || *activationRole != Role::activation_fp8)
      return ttatlas::emitUnsupported(
          matmul, "K-chain left operand requires an activation_fp8 slot");
  }
  if (!llvm::all_of(llvm::drop_begin(units), [&](int32_t unit) {
        return unit == units.front();
      }))
    return ttatlas::emitUnsupported(root,
                                    "one K chain must use a single MXU unit");

  Value state = latestStateBefore(root);
  if (!state) {
    Operation *parent = root->getParentOp();
    while (parent && parent->getName().getStringRef() != "tt.func" &&
           parent->getName().getStringRef() != "func.func")
      parent = parent->getParentOp();
    // The boundary pattern converts tt.func first; let dialect conversion
    // revisit this chain in the new func.func, where its state token exists.
    if (parent && parent->getName().getStringRef() == "tt.func")
      return failure();
    return ttatlas::emitUnsupported(
        root, "K-chain lowering requires the converted function state token");
  }
  if (!state.hasOneUse())
    return ttatlas::emitUnsupported(
        root, "K-chain state must have one successor to splice");
  OpOperand &successorUse = *state.getUses().begin();
  Operation *successor = successorUse.getOwner();
  if (successorUse.getOperandNumber() != 0 ||
      successor->getName().getDialectNamespace() != "atlas")
    return ttatlas::emitUnsupported(
        root, "K-chain state successor is not an Atlas stateful operation");

  rewriter.setInsertionPoint(root);
  MLIRContext *context = root->getContext();
  Type stateType = atlas::VirtualStateType::get(context);
  Type bf16Type = atlas::VirtualBF16Type::get(context);
  Value accumulator;
  for (auto [index, matmul] : llvm::enumerate(chain)) {
    Value activation = rewriter.getRemappedValue(matmul->getOperand(0));
    Value weight = rewriter.getRemappedValue(matmul->getOperand(1));
    if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
        transpose && transpose->getName().getStringRef() == "linalg.transpose")
      weight = rewriter.getRemappedValue(transpose->getOperand(0));
    if (roles[index] == Role::weight_fp8_kn) {
      Operation *transposed = createAtlasOp(
          rewriter, matmul->getLoc(),
          atlas::VirtualXLUTransposeOp::getOperationName(), ValueRange{weight},
          TypeRange{atlas::VirtualFP8Type::get(context)});
      weight = transposed->getResult(0);
    }

    Builder builder(context);
    Type weightType = atlas::VirtualMXUWeightType::get(context, units[index]);
    Operation *loadWeight = createAtlasOp(
        rewriter, matmul->getLoc(),
        atlas::VirtualMXULoadWeightOp::getOperationName(),
        ValueRange{state, weight}, TypeRange{stateType, weightType},
        ArrayRef<NamedAttribute>{builder.getNamedAttr(
            "unit", builder.getI32IntegerAttr(units[index]))});
    state = loadWeight->getResult(0);
    Operation *mxu;
    if (index == 0) {
      mxu = createAtlasOp(
          rewriter, matmul->getLoc(),
          atlas::VirtualMXUResetOp::getOperationName(),
          ValueRange{state, activation, loadWeight->getResult(1)},
          TypeRange{stateType, atlas::VirtualMXUAccType::get(context,
                                                              units[index])});
    } else {
      mxu = createAtlasOp(
          rewriter, matmul->getLoc(),
          atlas::VirtualMXUAccumulateOp::getOperationName(),
          ValueRange{state, activation, loadWeight->getResult(1), accumulator},
          TypeRange{stateType, atlas::VirtualMXUAccType::get(context,
                                                              units[index])});
    }
    state = mxu->getResult(0);
    accumulator = mxu->getResult(1);
  }

  Operation *readout = createAtlasOp(
      rewriter, chain.back()->getLoc(),
      atlas::VirtualMXUReadoutBF16Op::getOperationName(),
      ValueRange{state, accumulator},
      TypeRange{stateType, bf16Type});
  rewriter.modifyOpInPlace(successor, [&] {
    successor->setOperand(successorUse.getOperandNumber(),
                          readout->getResult(0));
  });

  Operation *terminal = chain.back();
  SmallVector<Operation *> truncs;
  for (Operation *user : terminal->getResult(0).getUsers())
    if (user->getName().getStringRef() == "arith.truncf" &&
        user->getNumResults() == 1)
      truncs.push_back(user);
  for (Operation *trunc : truncs)
    rewriter.replaceOp(trunc, readout->getResult(1));
  rewriter.replaceOp(terminal, readout->getResult(1));
  for (Operation *matmul : llvm::reverse(chain.drop_back()))
    rewriter.eraseOp(matmul);

  Operation *fill = root->getOperand(2).getDefiningOp();
  if (fill && fill->getName().getStringRef() == "linalg.fill") {
    Operation *scalar = fill->getOperand(0).getDefiningOp();
    Value destination = fill->getOperand(1);
    rewriter.eraseOp(fill);
    if (Operation *empty = destination.getDefiningOp();
        empty && empty->getName().getStringRef() == "tensor.empty" &&
        empty->getResult(0).use_empty())
      rewriter.eraseOp(empty);
    if (scalar && scalar->getName().getStringRef() == "arith.constant" &&
        llvm::all_of(scalar->getResults(), [](Value result) {
          return result.use_empty();
        }))
      rewriter.eraseOp(scalar);
  }
  return success();
}

// A linalg.transpose is redundant for the MXU's NK representation. Pass 6 put
// the role on its source load/slice, so inspect that source before dropping it.
struct RemoveNKTranspose : ConversionPattern {
  RemoveNKTranspose(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.transpose", PatternBenefit(2),
                          context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1 ||
        operands.empty())
      return failure();
    if (!llvm::all_of(op->getResult(0).getUses(), [&](OpOperand &use) {
          return use.getOwner()->getName().getStringRef() == "linalg.matmul" &&
                 use.getOperandNumber() == 1;
        }))
      return failure();
    auto role = getWeightRole(op->getOperand(0));
    if (!role || *role != Role::weight_fp8_nk)
      return failure();

    Value destination = op->getOperand(1);
    rewriter.replaceOp(op, operands.front());
    if (Operation *empty = destination.getDefiningOp();
        empty && empty->getName().getStringRef() == "tensor.empty" &&
        empty->getResult(0).use_empty())
      rewriter.eraseOp(empty);
    return success();
  }
};

struct ConvertMatmul : ConversionPattern {
  ConvertMatmul(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.matmul", PatternBenefit(1),
                          context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // K chains are handled separately below; a non-root member waits for its
    // first matmul, which owns the whole structurally matched chain.
    if (op->getNumOperands() < 3)
      return failure();
    Operation *init = op->getOperand(2).getDefiningOp();
    if (init && init->getName().getStringRef() == "linalg.matmul")
      return failure();
    if (!isZeroFill(init))
      return ttatlas::emitUnsupported(
          op, "matmul accumulator must start from linalg.fill zero");
    SmallVector<Operation *> chain = collectKChain(op);
    if (chain.empty())
      return ttatlas::emitUnsupported(op,
                                      "matmul has an unsupported K-chain use");
    if (chain.size() > 1)
      return lowerKChain(op, chain, rewriter);
    return replaceMatmul(op, operands, rewriter);
  }
};

} // namespace

void populateDotPatterns(TypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  patterns.add<RemoveNKTranspose, ConvertMatmul>(typeConverter,
                                                 patterns.getContext());
}

} // namespace mlir::triton
