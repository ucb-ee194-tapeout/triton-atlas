// Pass 7 boundary patterns for the IA-19 Linalg + TTIR pipeline.
// Boundary conversion owns the function ABI and the linear I/O state chain.
// Compute operations are cloned unchanged for the other registered patterns.
#include "Patterns.h"
#include "TritonToAtlasVirtual/Passes.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <map>

namespace mlir::triton {
namespace {

bool isAddressMath(Operation *op) {
  if (!isMemoryEffectFree(op) || op->getNumResults() != 1)
    return false;
  Type element = getElementTypeOrSelf(op->getResult(0).getType());
  return isa<IntegerType, PointerType>(element) &&
         isa<SplatOp, BroadcastOp, ExpandDimsOp, AddPtrOp, MakeRangeOp,
             arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp>(
             op);
}

struct ConvertBoundary : OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp source, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reject = [&](const Twine &reason) {
      ttatlas::emitUnsupported(source, reason);
      return failure();
    };
    if (source.isExternal() || source.getBody().getBlocks().size() != 1 ||
        source.getFunctionType().getNumResults() != 0)
      return reject("boundary conversion requires one defined, single-block, "
                    "void tt.func");
    Block &body = source.getBody().front();
    if (!isa<ReturnOp>(body.getTerminator()))
      return reject("expected tt.return at the kernel boundary");

    std::map<int32_t, std::pair<ttatlas::SlotAttr, Type>> inputs;
    SmallVector<int32_t> outputs;
    bool sawStore = false;
    for (Operation &op : body) {
      if (op.getNumRegions())
        return reject(
            "nested regions require the future control-flow boundary ABI");
      if (!isa<LoadOp, StoreOp>(op))
        continue;
      auto slot = op.getAttrOfType<ttatlas::SlotAttr>(ttatlas::kSlot);
      if (!slot)
        return reject("tt.load/tt.store is missing #ttatlas.slot from Pass 5");
      if (slot.getArg() >= source.getNumArguments() ||
          !isa<PointerType>(source.getArgument(slot.getArg()).getType()) ||
          slot.getLogicalShape() != ArrayRef<int64_t>{32, 32})
        return reject("slot argument or logical shape is incompatible with the "
                      "M1 boundary ABI");
      if (auto load = dyn_cast<LoadOp>(op)) {
        if (sawStore || load.getMask() || load.getIsVolatile())
          return reject("boundary inputs must be unmasked, nonvolatile loads "
                        "before all stores");
        auto tensor = dyn_cast<RankedTensorType>(load.getType());
        if (!tensor || tensor.getShape() != ArrayRef<int64_t>{32, 32} ||
            (!tensor.getElementType().isBF16() &&
             !isa<Float8E4M3FNType>(tensor.getElementType())) ||
            slot.getRole() == ttatlas::SlotRole::output_bf16)
          return reject("unsupported input tile type or slot role");
        auto [it, inserted] = inputs.emplace(
            slot.getIndex(), std::make_pair(slot, tensor.getElementType()));
        if (!inserted && (it->second.first != slot ||
                          it->second.second != tensor.getElementType()))
          return reject(
              "one input index names inconsistent slots or tile types");
      } else {
        auto store = cast<StoreOp>(op);
        auto tensor = dyn_cast<RankedTensorType>(store.getValue().getType());
        if (store.getMask() || !tensor ||
            tensor.getShape() != ArrayRef<int64_t>{32, 32} ||
            !tensor.getElementType().isBF16() ||
            slot.getRole() != ttatlas::SlotRole::output_bf16 ||
            llvm::is_contained(outputs, slot.getIndex()))
          return reject("outputs require unmasked BF16 tiles with unique "
                        "output_bf16 slots");
        sawStore = true;
        outputs.push_back(slot.getIndex());
      }
    }
    if (outputs.empty())
      return reject("an Atlas kernel requires at least one output boundary");
    int32_t expected = 0;
    for (const auto &[index, entry] : inputs)
      if (index != expected++)
        return reject("input indices must be dense from zero");
    llvm::sort(outputs);
    for (auto [index, actual] : llvm::enumerate(outputs))
      if (index != static_cast<size_t>(actual))
        return reject("output indices must be dense from zero");

    // Pointer arguments may only feed address math ending at I/O pointer
    // operands. No scalar/control ABI or escaping pointer is invented here.
    SmallVector<Value> pointers(source.getArguments());
    llvm::SmallPtrSet<Operation *, 32> visited;
    while (!pointers.empty()) {
      Value value = pointers.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isa<LoadOp, StoreOp>(user) && use.getOperandNumber() == 0)
          continue;
        if (!isAddressMath(user))
          return reject("function arguments may only feed supported boundary "
                        "address math");
        if (visited.insert(user).second)
          pointers.append(user->result_begin(), user->result_end());
      }
    }

    llvm::SmallPtrSet<Operation *, 32> addressOps;
    SmallVector<Value> addressValues;
    for (Operation &op : body) {
      if (auto load = dyn_cast<LoadOp>(op))
        addressValues.push_back(load.getPtr());
      if (auto store = dyn_cast<StoreOp>(op))
        addressValues.push_back(store.getPtr());
    }
    while (!addressValues.empty()) {
      Operation *def = addressValues.pop_back_val().getDefiningOp();
      if (!def)
        continue;
      if (!isAddressMath(def))
        return reject("unsupported boundary address operation");
      if (addressOps.insert(def).second)
        addressValues.append(def->operand_begin(), def->operand_end());
    }
    for (Operation *address : addressOps)
      for (OpOperand &use : address->getResult(0).getUses())
        if (!addressOps.contains(use.getOwner()) &&
            !(isa<LoadOp, StoreOp>(use.getOwner()) &&
              use.getOperandNumber() == 0))
          return reject("address values escape into compute operations");

    // Include otherwise-unused address expressions as well. Reverse order
    // proves that their only users are already in the discarded address slice.
    for (Operation &op : llvm::reverse(body))
      if (isAddressMath(&op) &&
          llvm::all_of(op.getResult(0).getUsers(), [&](Operation *user) {
            return addressOps.contains(user);
          }))
        addressOps.insert(&op);

    auto function = rewriter.create<func::FuncOp>(
        source.getLoc(), source.getName(),
        rewriter.getFunctionType(
            {}, {atlas::VirtualStateType::get(source.getContext())}));
    for (NamedAttribute attr : source->getAttrs())
      if (attr.getName() != "sym_name" && attr.getName() != "function_type" &&
          attr.getName() != "arg_attrs" && attr.getName() != "res_attrs" &&
          attr.getName() != "noinline")
        function->setAttr(attr.getName(), attr.getValue());
    Block *entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);
    auto stateType = atlas::VirtualStateType::get(source.getContext());
    OperationState start(source.getLoc(),
                         atlas::VirtualStartOp::getOperationName());
    start.addTypes(stateType);
    Value state = rewriter.create(start)->getResult(0);
    std::map<int32_t, Value> inputValues;
    for (const auto &[index, entry] : inputs) {
      auto [slot, element] = entry;
      bool bf16 = element.isBF16();
      OperationState input(source.getLoc(),
                           bf16 ? atlas::VirtualInputBF16Op::getOperationName()
                                : atlas::VirtualInputFP8Op::getOperationName());
      input.addOperands(state);
      Type tile = bf16 ? Type(atlas::VirtualBF16Type::get(source.getContext()))
                       : Type(atlas::VirtualFP8Type::get(source.getContext()));
      input.addTypes({stateType, tile});
      input.addAttribute("index", rewriter.getI32IntegerAttr(index));
      input.addAttribute(ttatlas::kSlot, slot);
      Operation *created = rewriter.create(input);
      state = created->getResult(0);
      inputValues[index] = created->getResult(1);
    }
    IRMapping mapping;
    for (Operation &op : body) {
      if (auto load = dyn_cast<LoadOp>(op)) {
        auto slot = op.getAttrOfType<ttatlas::SlotAttr>(ttatlas::kSlot);
        mapping.map(load.getResult(), inputValues.at(slot.getIndex()));
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        Value value = mapping.lookupOrDefault(store.getValue());
        // A compute pattern will convert this SSA result after the function
        // pattern. A tensor value is intentionally retained until that rewrite.
        auto slot = op.getAttrOfType<ttatlas::SlotAttr>(ttatlas::kSlot);
        OperationState output(op.getLoc(),
                              atlas::VirtualOutputBF16Op::getOperationName());
        output.addOperands({state, value});
        output.addTypes(stateType);
        output.addAttribute("index",
                            rewriter.getI32IntegerAttr(slot.getIndex()));
        output.addAttribute(ttatlas::kSlot, slot);
        state = rewriter.create(output)->getResult(0);
      } else if (isa<ReturnOp>(op)) {
        rewriter.create<func::ReturnOp>(op.getLoc(), state);
      } else if (!addressOps.contains(&op)) {
        rewriter.clone(op, mapping);
      }
    }
    rewriter.eraseOp(source);
    return success();
  }
};
} // namespace

void populateBoundaryPatterns(TypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  patterns.add<ConvertBoundary>(typeConverter, patterns.getContext());
}
} // namespace mlir::triton
