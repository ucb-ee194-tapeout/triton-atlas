// Pass 7 lowering for MXU matmuls. The ordinary two-tile form is a virtual
// matmul; multi-K forms become an explicitly ordered MXU resource chain.
#include "Patterns.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::triton {
namespace {

using Role = ttatlas::SlotRole;

static std::optional<Role> getWeightRole(Value value) {
  llvm::SmallPtrSet<Operation *, 8> seen;
  while (Operation *definition = value.getDefiningOp()) {
    if (!seen.insert(definition).second)
      return std::nullopt;
    if (auto slot =
            definition->getAttrOfType<ttatlas::SlotAttr>(ttatlas::kSlot))
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

// A direct Atlas input gets its role from its slot.  An intermediate BF16
// tile that is packed to FP8 has no DRAM slot because it stays on Atlas.  It
// can be an activation (an MLP handoff) or a weight-like input (Q/K/V
// projections in the SmolVLA transformer core).
static bool isPackedFP8Tile(Value value) {
  auto generic = value.getDefiningOp<linalg::GenericOp>();
  if (!generic || generic.getNumDpsInputs() != 1)
    return false;
  auto type = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
  if (!type || !isa<Float8E4M3FNType>(type.getElementType()))
    return false;
  // Pass 3 hoists the scalar conversion's scale tag to this generic.
  return generic->hasAttr(ttatlas::kScale);
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

// Unlike a transposed DRAM weight (which eraseNKTransposes handles before
// conversion), a transposed on-chip K tile has real semantics.  Lower it to
// the FP8-only XLU operation so attention can use Q x K^T.
struct ConvertFP8Transpose : ConversionPattern {
  ConvertFP8Transpose(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.transpose", PatternBenefit(2),
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1 ||
        operands.empty())
      return failure();
    auto permutation = op->getAttrOfType<DenseI64ArrayAttr>("permutation");
    if (!permutation || permutation.asArrayRef() != ArrayRef<int64_t>{1, 0})
      return ttatlas::emitUnsupported(op,
                                      "only the FP8 [1, 0] transpose is supported");
    if (!isa<atlas::VirtualFP8Type>(operands.front().getType()) ||
        !isa<atlas::VirtualFP8Type>(
            getTypeConverter()->convertType(op->getResult(0).getType())))
      return ttatlas::emitUnsupported(op, "Atlas XLU transposes FP8 tiles only");
    Operation *lowered = createAtlasOp(
        rewriter, op->getLoc(), atlas::VirtualXLUTransposeOp::getOperationName(),
        ValueRange{operands.front()}, TypeRange{operands.front().getType()});
    Operation *empty = op->getNumOperands() > 1
                           ? op->getOperand(1).getDefiningOp()
                           : nullptr;
    rewriter.replaceOp(op, lowered->getResults());
    if (empty && empty->getName().getStringRef() == "tensor.empty" &&
        empty->use_empty())
      rewriter.eraseOp(empty);
    return success();
  }
};

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

static LogicalResult replaceMatmul(Operation *matmul, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) {
  if (matmul->getNumOperands() != 3 || operands.size() != 3 ||
      matmul->getNumResults() != 1)
    return ttatlas::emitUnsupported(matmul,
                                    "expected a single-result linalg.matmul");

  FailureOr<int32_t> unit = getMxuUnit(matmul);
  if (failed(unit))
    return failure();
  std::optional<Role> role = getWeightRole(matmul->getOperand(1));
  Operation *logicalTranspose = matmul->getOperand(1).getDefiningOp();
  bool packedWeight = isPackedFP8Tile(matmul->getOperand(1)) ||
                      (logicalTranspose &&
                       logicalTranspose->getName().getStringRef() ==
                           "linalg.transpose" &&
                       isPackedFP8Tile(logicalTranspose->getOperand(0)));
  if ((!role || (*role != Role::weight_fp8_nk &&
                 *role != Role::weight_fp8_kn)) &&
      !packedWeight)
    return ttatlas::emitUnsupported(
        matmul,
        "matmul weight requires a weight_fp8_nk/weight_fp8_kn slot or a "
        "packed on-chip FP8 tile");
  if (role && *role == Role::weight_fp8_kn)
    if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
        transpose && transpose->getName().getStringRef() == "linalg.transpose")
      return ttatlas::emitUnsupported(
          matmul, "weight_fp8_kn must not have a linalg.transpose producer");
  auto activationRole = getWeightRole(matmul->getOperand(0));
  if ((!activationRole || *activationRole != Role::activation_fp8) &&
      !isPackedFP8Tile(matmul->getOperand(0)))
    return ttatlas::emitUnsupported(
        matmul,
        "matmul left operand requires an activation_fp8 slot or a packed "
        "result from the preceding MLP layer");
  // Let ConvertFP8Transpose produce the XLU result first.  It avoids
  // duplicating the transpose when this matmul is revisited by conversion.
  if (packedWeight && logicalTranspose &&
      logicalTranspose->getName().getStringRef() == "linalg.transpose" &&
      !isa<atlas::VirtualFP8Type>(operands[1].getType()))
    return failure();

  Operation *fill = matmul->getOperand(2).getDefiningOp();
  // ConvertMatmul already checked the structural zero-fill seed before it
  // dispatched here.  During dialect conversion, the seed can temporarily
  // have an additional bookkeeping use while its consumer is being replaced;
  // do not reject that transient state a second time.  A genuinely shared
  // seed leaves an illegal linalg user behind and therefore still fails the
  // full conversion.
  if (!isZeroFill(fill))
    return ttatlas::emitUnsupported(
        matmul, "matmul accumulator must come from a linalg.fill zero");

  Value activation = operands[0];
  Value weight = operands[1];
  if (role && *role == Role::weight_fp8_nk)
    if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
        transpose && transpose->getName().getStringRef() == "linalg.transpose")
      weight = rewriter.getRemappedValue(transpose->getOperand(0));
  if (role && *role == Role::weight_fp8_kn) {
    Type fp8 = atlas::VirtualFP8Type::get(matmul->getContext());
    Operation *transpose =
        createAtlasOp(rewriter, matmul->getLoc(),
                      atlas::VirtualXLUTransposeOp::getOperationName(),
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
  // The zero fill stays: other matmuls may share it (CSE merges equal fills),
  // and the driver sweeps it once nothing reads it.
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

// A one-input, all-parallel linalg.generic whose body only truncates its f32
// input to BF16: pass 5's marker of where a matmul chain is read out.
static bool isReadoutMarker(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1)
    return false;
  Block &body = generic.getRegion().front();
  return body.getOperations().size() == 2 && isa<arith::TruncFOp>(body.front());
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
    if (!role || (*role != Role::weight_fp8_nk && *role != Role::weight_fp8_kn))
      return ttatlas::emitUnsupported(
          matmul,
          "K-chain weight requires a weight_fp8_nk or weight_fp8_kn slot");
    roles.push_back(*role);
    if (*role == Role::weight_fp8_kn)
      if (Operation *transpose = matmul->getOperand(1).getDefiningOp();
          transpose &&
          transpose->getName().getStringRef() == "linalg.transpose")
        return ttatlas::emitUnsupported(
            matmul, "weight_fp8_kn must not have a linalg.transpose producer");
    auto activationRole = getWeightRole(matmul->getOperand(0));
    if (!activationRole || *activationRole != Role::activation_fp8)
      return ttatlas::emitUnsupported(
          matmul, "K-chain left operand requires an activation_fp8 slot");
  }
  if (!llvm::all_of(llvm::drop_begin(units),
                    [&](int32_t unit) { return unit == units.front(); }))
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
          TypeRange{stateType,
                    atlas::VirtualMXUAccType::get(context, units[index])});
    } else {
      mxu = createAtlasOp(
          rewriter, matmul->getLoc(),
          atlas::VirtualMXUAccumulateOp::getOperationName(),
          ValueRange{state, activation, loadWeight->getResult(1), accumulator},
          TypeRange{stateType,
                    atlas::VirtualMXUAccType::get(context, units[index])});
    }
    state = mxu->getResult(0);
    accumulator = mxu->getResult(1);
  }

  Operation *readout = createAtlasOp(
      rewriter, chain.back()->getLoc(),
      atlas::VirtualMXUReadoutBF16Op::getOperationName(),
      ValueRange{state, accumulator}, TypeRange{stateType, bf16Type});
  rewriter.modifyOpInPlace(successor, [&] {
    successor->setOperand(successorUse.getOperandNumber(),
                          readout->getResult(0));
  });

  Operation *terminal = chain.back();
  // The readout marker pass 5 puts after a chain (a linalg.generic that only
  // narrows f32 to BF16), or a bare tile truncf, is the value this readout
  // produces.
  SmallVector<Operation *> truncs;
  for (Operation *user : terminal->getResult(0).getUsers())
    if ((user->getName().getStringRef() == "arith.truncf" ||
         isReadoutMarker(user)) &&
        user->getNumResults() == 1)
      truncs.push_back(user);
  for (Operation *trunc : truncs)
    rewriter.replaceOp(trunc, readout->getResult(1));
  rewriter.replaceOp(terminal, readout->getResult(1));
  for (Operation *matmul : llvm::reverse(chain.drop_back()))
    rewriter.eraseOp(matmul);

  return success();
}

struct ConvertMatmul : ConversionPattern {
  ConvertMatmul(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.matmul", PatternBenefit(1),
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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

// A linalg.transpose of a weight_fp8_nk tile is redundant: the MXU takes the NK
// layout as it is. Pass 6 put the role on the source load. Done before the
// conversion, since the boundary conversion feeds the cloned ops virtual
// tiles, which a transpose's folding (tried before any pattern) cannot read.
void eraseNKTransposes(ModuleOp module) {
  SmallVector<Operation *> transposes;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "linalg.transpose" ||
        op->getNumOperands() < 2 || op->getNumResults() != 1)
      return;
    auto permutation = op->getAttrOfType<DenseI64ArrayAttr>("permutation");
    auto role = getWeightRole(op->getOperand(0));
    if (permutation && permutation.asArrayRef() == ArrayRef<int64_t>{1, 0} &&
        role && *role == Role::weight_fp8_nk &&
        llvm::all_of(op->getResult(0).getUses(), [](OpOperand &use) {
          return use.getOwner()->getName().getStringRef() == "linalg.matmul" &&
                 use.getOperandNumber() == 1;
        }))
      transposes.push_back(op);
  });
  for (Operation *op : transposes) {
    Value destination = op->getOperand(1);
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->erase();
    if (Operation *empty = destination.getDefiningOp();
        empty && empty->getName().getStringRef() == "tensor.empty" &&
        empty->use_empty())
      empty->erase();
  }
}

void populateDotPatterns(TypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  patterns.add<ConvertFP8Transpose, ConvertMatmul>(typeConverter,
                                                    patterns.getContext());
}

} // namespace mlir::triton
