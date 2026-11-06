#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_ASSIGNSLOTS
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {
using Coordinates = SmallVector<int64_t, 2>;

// Evaluate only the explicitly supported integer grammar. APInt preserves TTIR
// integer wrapping semantics (including negative constants), rather than using
// unchecked host arithmetic. No divisibility/contiguity hints are trusted.
std::optional<llvm::APInt> integerAt(Value value, Coordinates at,
                                     unsigned depth = 0) {
  if (depth > 128)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto c = dyn_cast<arith::ConstantOp>(op)) {
    if (auto scalar = dyn_cast<IntegerAttr>(c.getValue()))
      return scalar.getValue();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue())) {
      auto shape = dense.getType().getShape();
      if (shape.size() != at.size())
        return std::nullopt;
      uint64_t index = 0;
      for (auto [size, coordinate] : llvm::zip(shape, at))
        index = index * size + coordinate;
      return *(dense.getValues<llvm::APInt>().begin() + index);
    }
  }
  if (auto range = dyn_cast<MakeRangeOp>(op)) {
    if (at.size() == 1)
      return llvm::APInt(32, range.getStartAttr().getInt() + at[0]);
    return std::nullopt;
  }
  if (isa<SplatOp>(op))
    return integerAt(op->getOperand(0), {}, depth + 1);
  if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
    at.erase(at.begin() + expand.getAxis());
    return integerAt(expand.getSrc(), at, depth + 1);
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
    auto shape =
        cast<RankedTensorType>(broadcast.getSrc().getType()).getShape();
    for (unsigned i = 0; i < at.size(); ++i)
      if (shape[i] == 1)
        at[i] = 0;
    return integerAt(broadcast.getSrc(), at, depth + 1);
  }
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
    auto lhs = integerAt(op->getOperand(0), at, depth + 1);
    auto rhs = integerAt(op->getOperand(1), at, depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    bool signedOverflow, unsignedOverflow;
    llvm::APInt result = *lhs;
    if (isa<arith::AddIOp>(op)) {
      result = lhs->sadd_ov(*rhs, signedOverflow);
      (void)lhs->uadd_ov(*rhs, unsignedOverflow);
    } else if (isa<arith::SubIOp>(op)) {
      result = lhs->ssub_ov(*rhs, signedOverflow);
      (void)lhs->usub_ov(*rhs, unsignedOverflow);
    } else {
      result = lhs->smul_ov(*rhs, signedOverflow);
      (void)lhs->umul_ov(*rhs, unsignedOverflow);
    }
    auto flags = cast<arith::ArithIntegerOverflowFlagsInterface>(op);
    if ((flags.hasNoSignedWrap() && signedOverflow) ||
        (flags.hasNoUnsignedWrap() && unsignedOverflow))
      return std::nullopt;
    return result;
  }
  return std::nullopt;
}

struct PointerLane {
  BlockArgument argument;
  int64_t offset;
};

std::optional<PointerLane> pointerAt(Value value, Coordinates at, FuncOp func,
                                     unsigned depth = 0) {
  if (depth > 128)
    return std::nullopt;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == &func.getBody().front() &&
        isa<PointerType>(arg.getType()))
      return PointerLane{arg, 0};
    return std::nullopt;
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (isa<SplatOp>(op))
    return pointerAt(op->getOperand(0), {}, func, depth + 1);
  if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
    at.erase(at.begin() + expand.getAxis());
    return pointerAt(expand.getSrc(), at, func, depth + 1);
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
    auto shape =
        cast<RankedTensorType>(broadcast.getSrc().getType()).getShape();
    for (unsigned i = 0; i < at.size(); ++i)
      if (shape[i] == 1)
        at[i] = 0;
    return pointerAt(broadcast.getSrc(), at, func, depth + 1);
  }
  if (auto add = dyn_cast<AddPtrOp>(op)) {
    auto base = pointerAt(add.getPtr(), at, func, depth + 1);
    auto offset = integerAt(add.getOffset(), at, depth + 1);
    if (!base || !offset || offset->getBitWidth() > 64)
      return std::nullopt;
    bool overflow;
    auto sum = llvm::APInt(64, base->offset)
                   .sadd_ov(offset->sextOrTrunc(64), overflow);
    if (overflow)
      return std::nullopt;
    return PointerLane{base->argument, sum.getSExtValue()};
  }
  return std::nullopt;
}

// Derive FP8 operand orientation from actual dot uses. A transposed source
// consumed as B is host NK; a direct B operand is already KN.
std::optional<ttatlas::SlotRole> roleFor(LoadOp load) {
  using Role = ttatlas::SlotRole;
  SmallVector<std::pair<Value, bool>> pending{{load.getResult(), false}};
  SmallVector<Role> roles;
  unsigned visits = 0;
  bool bf16 = getElementTypeOrSelf(load.getType()).isBF16();
  while (!pending.empty()) {
    if (++visits > 256)
      return std::nullopt;
    auto [value, transposed] = pending.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<DotOp>(user)) {
        if (use.getOperandNumber() == 0 && !transposed)
          roles.push_back(Role::activation_fp8);
        else if (use.getOperandNumber() == 1)
          roles.push_back(transposed ? Role::weight_fp8_nk
                                     : Role::weight_fp8_kn);
        else
          return std::nullopt;
      } else if (auto trans = dyn_cast<TransOp>(user)) {
        if (trans.getOrder() != ArrayRef<int32_t>{1, 0})
          return std::nullopt;
        pending.emplace_back(trans.getResult(), !transposed);
      } else if (isa<arith::ExtFOp, arith::TruncFOp>(user)) {
        pending.emplace_back(user->getResult(0), transposed);
      } else if (bf16 && isa<arith::AddFOp>(user)) {
        roles.push_back(Role::bias_bf16);
      }
    }
  }
  if (roles.empty())
    return bf16 ? std::optional<Role>(Role::constant_bf16) : std::nullopt;
  if (!llvm::all_of(roles, [&](Role role) { return role == roles.front(); }))
    return std::nullopt;
  return roles.front();
}

struct AssignedSlot {
  Operation *op;
  ttatlas::SlotAttr slot;
};

struct AssignSlots : impl::AssignSlotsBase<AssignSlots> {
  void runOnOperation() override {
    SmallVector<AssignedSlot> pending;
    bool invalid = false;
    for (FuncOp func : getOperation().getOps<FuncOp>()) {
      int64_t nextInput = 0, nextOutput = 0;
      bool sawStore = false;
      SmallVector<ttatlas::SlotAttr> inputSlots;
      func.walk([&](Operation *op) {
        if (!isa<LoadOp, StoreOp>(op))
          return;
        auto reject = [&](const Twine &reason) {
          ttatlas::emitUnsupported(op, reason);
          invalid = true;
        };
        if (func.getBody().getBlocks().size() != 1 ||
            op->getBlock() != &func.getBody().front()) {
          reject("slot assignment requires a single straight-line entry block");
          return;
        }
        auto load = dyn_cast<LoadOp>(op);
        auto store = dyn_cast<StoreOp>(op);
        Value ptr = load ? load.getPtr() : store.getPtr();
        Value mask = load ? load.getMask() : store.getMask();
        if (mask) {
          reject("masked accesses are unsupported; a whole unmasked tile is "
                 "required");
          return;
        }
        if (load && load.getIsVolatile()) {
          reject("volatile loads cannot be represented by an input slot");
          return;
        }
        auto host = op->getAttrOfType<TypeAttr>(ttatlas::kHostDtype);
        if (!host) {
          reject("missing ttatlas.host_dtype from Pass 1");
          return;
        }
        auto tensor = dyn_cast<RankedTensorType>(ptr.getType());
        if (!tensor || tensor.getShape() != ArrayRef<int64_t>{32, 32}) {
          reject("expected a static 32x32 tensor of pointers; 1-D bias needs "
                 "the Stage B driver-broadcast contract");
          return;
        }
        auto pointer = dyn_cast<PointerType>(tensor.getElementType());
        if (!pointer) {
          reject("tensor/block pointers are unsupported");
          return;
        }
        Type element = pointer.getPointeeType();
        if (!element.isBF16() && !isa<Float8E4M3FNType>(element)) {
          reject("expected legalized BF16 or FP8 E4M3FN elements");
          return;
        }
        if (store && !element.isBF16()) {
          reject("FP8 stores have no Atlas virtual output boundary operation");
          return;
        }
        SmallVector<int64_t> logicalShape{32, 32};
        auto pad = op->getAttrOfType<ttatlas::PadAttr>(ttatlas::kPad);
        if (store && !pad)
          if (Operation *producer = store.getValue().getDefiningOp())
            pad = producer->getAttrOfType<ttatlas::PadAttr>(ttatlas::kPad);
        if (pad) {
          if (pad.getLogicalShape() != ArrayRef<int64_t>{32, 32}) {
            reject(
                "partial padded boundaries require Stage B packing semantics");
            return;
          }
          logicalShape.assign(pad.getLogicalShape().begin(),
                              pad.getLogicalShape().end());
        }
        if (load && sawStore) {
          reject("loads after stores require memory-dependence analysis");
          return;
        }
        sawStore |= bool(store);
        auto origin = pointerAt(ptr, {0, 0}, func);
        if (!origin) {
          reject("cannot prove a static pointer expression rooted in a "
                 "function pointer argument (dynamic strides/offsets or "
                 "unsupported transformations)");
          return;
        }
        if (origin->offset < 0 ||
            origin->offset > std::numeric_limits<int64_t>::max() - 1023 ||
            origin->offset % 32) {
          reject("tile offset must be nonnegative, row-aligned, and fit signed "
                 "i64");
          return;
        }
        for (int64_t row = 0; row < 32; ++row)
          for (int64_t col = 0; col < 32; ++col) {
            auto lane = pointerAt(ptr, {row, col}, func);
            if (!lane || lane->argument != origin->argument ||
                lane->offset != origin->offset + row * 32 + col) {
              reject("cannot prove contiguous row-major tile: expected offset "
                     "+ row * 32 + column at every lane");
              return;
            }
          }
        auto role = load ? roleFor(load)
                         : std::optional<ttatlas::SlotRole>(
                               ttatlas::SlotRole::output_bf16);
        if (!role) {
          reject("cannot infer a unique activation/weight role from dot uses");
          return;
        }
        if (pad && load && pad.getNeutral() == ttatlas::PadNeutral::neg_inf)
          role = ttatlas::SlotRole::mask_bf16;
        if (*role == ttatlas::SlotRole::mask_bf16 && !element.isBF16()) {
          reject("a mask slot requires BF16 elements");
          return;
        }
        MLIRContext *context = op->getContext();
        auto tileOrigin = AffineMapAttr::get(
            AffineMap::get(0, 6,
                           {getAffineConstantExpr(origin->offset / 32, context),
                            getAffineConstantExpr(0, context)},
                           context));
        int64_t index = load ? nextInput : nextOutput;
        // Input reuse includes source argument, exact origin, role, shape and
        // original host dtype. Stores remain distinct ordered output events.
        for (ttatlas::SlotAttr previous : inputSlots)
          if (load && previous.getArg() == origin->argument.getArgNumber() &&
              previous.getTileOrigin() == tileOrigin &&
              previous.getRole() == *role &&
              previous.getLogicalShape() == ArrayRef<int64_t>(logicalShape) &&
              previous.getHostDtype() == host.getValue()) {
            index = previous.getIndex();
            break;
          }
        if (index > std::numeric_limits<int32_t>::max()) {
          reject("slot index exceeds Atlas signed i32 range");
          return;
        }
        auto emitError = [&]() { return op->emitError(); };
        auto slot = ttatlas::SlotAttr::getChecked(
            llvm::function_ref<InFlightDiagnostic()>(emitError), context,
            static_cast<int32_t>(index),
            static_cast<int32_t>(origin->argument.getArgNumber()), *role,
            tileOrigin, ArrayRef<int64_t>(logicalShape), host.getValue());
        if (!slot) {
          invalid = true;
          return;
        }
        if (load && index == nextInput) {
          ++nextInput;
          inputSlots.push_back(slot);
        }
        if (store)
          ++nextOutput;
        pending.push_back({op, slot});
      });
    }
    if (invalid) {
      signalPassFailure();
      return;
    }
    for (const AssignedSlot &assigned : pending)
      assigned.op->setAttr(ttatlas::kSlot, assigned.slot);
  }
};
} // namespace
} // namespace mlir::triton
