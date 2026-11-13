// Pass 1's conditions on legal ops (the "accepted only ..." rows of the op
// inventory). Each check reports its reason through `emit` when given one, the
// way MLIR verifiers do, so one function answers both "is it legal?" (the
// ConversionTarget's callback) and "why not?" (the error).
#include "ValidateTTIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::validate {

namespace {

LogicalResult fail(EmitFn emit, const Twine &reason) {
  if (emit)
    emit() << reason;
  return failure();
}

std::string printed(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  return text;
}

// 1.0, as a constant or a splat of one.
bool isOne(Value value) {
  if (auto splat = value.getDefiningOp<SplatOp>())
    value = splat.getSrc();
  return matchPattern(value, m_OneFloat());
}

// Whether a value only feeds tl.assume: llvm.intr.assume directly, or the
// arith.cmpi that builds its condition. The inventory drops the assume.
bool onlyFeedsAssumes(Operation *op) {
  return llvm::all_of(op->getUsers(), [](Operation *user) {
    if (isa<LLVM::AssumeOp>(user))
      return true;
    return isa<arith::CmpIOp>(user) && onlyFeedsAssumes(user);
  });
}

// Scalar arguments may be unused (Inductor passes sizes it doesn't read) or
// only feed a tl.assume.
// - atlas-hw: vpu_splat takes its value as constant bits, so a float has no
//   path into Atlas as data.
// - An integer (a size, a stride, a loop bound) needs specialization first:
//   the pass 1 page's subset has constant sizes and strides only, and no pass
//   evaluates a runtime offset or trip count yet.
LogicalResult checkFunc(FuncOp func, EmitFn emit) {
  for (BlockArgument arg : func.getArguments()) {
    bool usedOutsideAssumes = llvm::any_of(arg.getUsers(), [](Operation *user) {
      return !isa<arith::CmpIOp>(user) || !onlyFeedsAssumes(user);
    });
    if (!usedOutsideAssumes)
      continue;
    std::string which = "uses the runtime scalar argument #" +
                        std::to_string(arg.getArgNumber());
    if (isa<FloatType>(arg.getType()))
      return fail(emit, which + " as data; vpu_splat needs constant bits");
    if (isa<IntegerType>(arg.getType()))
      return fail(emit, which + " (a size, stride or bound); only constant "
                                "sizes and strides until specialization "
                                "exists");
  }
  return success();
}

LogicalResult checkLoad(LoadOp load, EmitFn emit) {
  if (!isa<RankedTensorType>(load.getType()))
    return fail(emit, "a scalar load; Atlas loads whole tiles");
  Type dtype = pointee(load.getPtr());
  if (dtype.isF32())
    return fail(emit, "an f32 source; Atlas has no f32 datapath and no "
                      "precision policy allows one yet");
  if (!isa<Float8E4M3FNType>(dtype) && !dtype.isBF16())
    return fail(emit, "loads " + printed(dtype) +
                          "; Atlas loads FP8 E4M3 or BF16 only");
  return success();
}

LogicalResult checkStore(StoreOp store, EmitFn emit) {
  if (!isa<RankedTensorType>(store.getValue().getType()))
    return fail(emit, "a scalar store; Atlas stores whole tiles");
  Type dtype = pointee(store.getPtr());
  if (!dtype.isBF16() && !dtype.isF32())
    return fail(emit, "stores " + printed(dtype) +
                          "; Atlas writes BF16, which the driver converts to "
                          "f32 only");
  return success();
}

// BF16 to f32 or back, plus BF16 to FP8 E4M3 for the output of one MLP layer
// feeding the next.  The first MLP policy uses Atlas's unit scale (127): this
// makes the choice explicit in pass 3, where the conversion becomes a pack.
// Narrowing must round to nearest even: that is what arith.truncf (pass 3)
// and Atlas do, so another tt.fp_to_fp rounding would change results
// silently.
LogicalResult checkConversion(Operation *op, EmitFn emit) {
  Type from = getElementTypeOrSelf(op->getOperand(0).getType());
  Type to = getElementTypeOrSelf(op->getResult(0).getType());
  bool bf16ToFp8 = from.isBF16() && isa<Float8E4M3FNType>(to);
  if (!(from.isBF16() && to.isF32()) && !(from.isF32() && to.isBF16()) &&
      !bf16ToFp8)
    return fail(emit, "converts " + printed(from) + " to " + printed(to) +
                          "; only BF16 <-> f32 or BF16 -> FP8 E4M3 is "
                          "accepted");
  bool narrowing = (from.isF32() && to.isBF16()) || bf16ToFp8;
  if (auto fpToFp = dyn_cast<FpToFpOp>(op); fpToFp && narrowing)
    if (std::optional<RoundingMode> rounding = fpToFp.getRounding();
        rounding && *rounding != RoundingMode::RTNE)
      return fail(emit, "narrows with rounding " +
                            stringifyRoundingMode(*rounding) +
                            "; Atlas rounds to nearest even (rtne) only");
  return success();
}

LogicalResult checkDivF(arith::DivFOp divf, EmitFn emit) {
  if (isOne(divf.getLhs()))
    return success();
  return fail(emit, "only 1 / x is accepted (Atlas's reciprocal table); "
                    "x / y needs a rewrite that doesn't exist yet");
}

// Whether a tt.reduce's body is one combine Atlas's VPU reduces a row with:
// add, max or min of the two block arguments, returned as is (atlas-hw:
// virtual IR's row reductions, per the pass 1 page's op inventory).
bool isRowCombine(ReduceOp reduce) {
  Block &body = reduce.getCombineOp().front();
  if (body.getOperations().size() != 2)
    return false;
  Operation *combine = &body.front();
  // unverified: maximumf/minimumf (NaN wins) and maxnumf/minnumf (number
  // wins) are both accepted; which one the VPU's max matches isn't checked.
  if (!isa<arith::AddFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
           arith::MinimumFOp>(combine))
    return false;
  auto args = body.getArguments();
  bool onArgs =
      (combine->getOperand(0) == args[0] &&
       combine->getOperand(1) == args[1]) ||
      (combine->getOperand(0) == args[1] && combine->getOperand(1) == args[0]);
  return onArgs && body.getTerminator()->getOperand(0) == combine->getResult(0);
}

LogicalResult checkReduce(ReduceOp reduce, EmitFn emit) {
  auto type = cast<RankedTensorType>(reduce.getSrcs()[0].getType());
  if (type.getRank() != 2 || reduce.getAxis() != 1)
    return fail(emit, "reduces axis " + Twine(reduce.getAxis()) + " of a " +
                          Twine(type.getRank()) +
                          "-D tensor; Atlas reduces rows of a 2-D tile only "
                          "(axis 1)");
  if (reduce.getNumOperands() != 1 || !isRowCombine(reduce))
    return fail(emit,
                "the combine is not one float add, max or min of one tensor");
  return success();
}

// atlas-hw: the MXU multiplies FP8 E4M3 tiles; an FP8 operand can only come
// from a load, since conversion to FP8 is illegal. Triton's dot accumulates in
// f32; pass 5 keeps that type as a label for the MXU accumulator (BF16 in
// hardware, rounded after every step on MXU0) and reads it out as BF16.
LogicalResult checkDot(DotOp dot, EmitFn emit) {
  for (Value operand : {dot.getA(), dot.getB()}) {
    Type element = getElementTypeOrSelf(operand.getType());
    if (!isa<Float8E4M3FNType>(element))
      return fail(emit, "multiplies " + printed(element) +
                            "; Atlas's MXU takes FP8 E4M3 only");
  }
  Type result = getElementTypeOrSelf(dot.getType());
  if (!result.isF32())
    return fail(emit, "accumulates in " + printed(result) +
                          "; pass 5 expects Triton's f32 dot result");
  return success();
}

// atlas-hw: the XLU transposes FP8 only, and only as a matmul weight (the
// dot's second operand; slot role weight_fp8_kn).
LogicalResult checkTrans(TransOp trans, EmitFn emit) {
  Type element = getElementTypeOrSelf(trans.getType());
  if (!isa<Float8E4M3FNType>(element))
    return fail(emit, "transposes " + printed(element) +
                          "; Atlas transposes FP8 only");
  for (OpOperand &use : trans->getUses())
    if (!isa<DotOp>(use.getOwner()) || use.getOperandNumber() != 1)
      return fail(emit, "the transpose feeds something other than a tt.dot's "
                        "second operand; Atlas transposes FP8 weights only");
  return success();
}

LogicalResult checkPoison(ub::PoisonOp poison, EmitFn emit) {
  if (poison->use_empty())
    return success();
  return fail(emit, "a used poison value has no Atlas meaning");
}

// Pass 2 unrolls the loop fully, which needs a constant trip count.
LogicalResult checkFor(scf::ForOp loop, EmitFn emit) {
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!getConstantIntValue(loop.getLowerBound()) ||
      !getConstantIntValue(loop.getUpperBound()) || !step)
    return fail(emit, "runtime trip count; pass 2 unrolls constant-trip loops "
                      "only (atlas-mlir accepts only constant DMA addresses)");
  if (*step <= 0)
    return fail(emit,
                "the step is not positive, so the loop can't be unrolled");
  return success();
}

} // namespace

LogicalResult checkConditions(Operation *op, EmitFn emit) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](FuncOp func) { return checkFunc(func, emit); })
      .Case([&](LoadOp load) { return checkLoad(load, emit); })
      .Case([&](StoreOp store) { return checkStore(store, emit); })
      .Case<FpToFpOp, arith::ExtFOp, arith::TruncFOp>(
          [&](Operation *conversion) {
            return checkConversion(conversion, emit);
          })
      .Case([&](DotOp dot) { return checkDot(dot, emit); })
      .Case([&](arith::DivFOp divf) { return checkDivF(divf, emit); })
      .Case([&](ReduceOp reduce) { return checkReduce(reduce, emit); })
      .Case([&](TransOp trans) { return checkTrans(trans, emit); })
      .Case([&](ub::PoisonOp poison) { return checkPoison(poison, emit); })
      .Case([&](scf::ForOp loop) { return checkFor(loop, emit); })
      .Default([](Operation *) { return success(); });
}

} // namespace mlir::triton::validate
