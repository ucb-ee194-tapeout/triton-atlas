// Pass 1's op inventory (triton-atlas design agenda, page "Pass 1 ·
// ttatlas-validate-ttir", section "Op inventory") as a ConversionTarget. Every
// op TTIR can contain is listed once: 175 ops, tt from Triton v3.9.0's
// TritonOps.td; arith, math, scf, cf and ub from the LLVM Triton 3.9.0 builds
// against (b010a18d); plus gpu.barrier, llvm.intr.assume and builtin.module.
//
// "Legal" means some later pass lowers the op today. When a pass learns a new
// case, its row moves to a legal verdict here in the same change.
#include "ValidateTTIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::validate {

namespace {

// One kind of illegal verdict. reject<Ops...>(detail) marks the ops illegal
// and records each one's error text, "<kind>: <detail>".
struct Rejection {
  ConversionTarget &target;
  Reasons &reasons;
  StringRef kind;

  template <typename... Ops> void reject(StringRef detail) const {
    target.addIllegalOp<Ops...>();
    std::string reason = (kind + ": " + detail).str();
    auto record = [&](StringRef name) {
      bool inserted =
          reasons.try_emplace(OperationName(name, &target.getContext()), reason)
              .second;
      assert(inserted && "op listed twice in the inventory");
      (void)inserted;
    };
    (record(Ops::getOperationName()), ...);
  }
};

// Lowered as they are.
void addLegal(ConversionTarget &target) {
  target.addLegalOp<ModuleOp, ReturnOp, MakeRangeOp, SplatOp, BroadcastOp,
                    ExpandDimsOp, AddPtrOp, ReduceReturnOp, GetProgramIdOp,
                    GetNumProgramsOp>();
  // unverified: math.exp and math.exp2 go to Atlas's exp table, whose
  // accuracy hasn't been checked on hardware.
  target.addLegalOp<arith::ConstantOp, arith::AddFOp, arith::SubFOp,
                    arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp,
                    arith::MaximumFOp, arith::MinimumFOp, arith::AddIOp,
                    arith::SubIOp, arith::MulIOp, math::ExpOp, math::Exp2Op,
                    scf::YieldOp>();
  // Masks: pass 4 turns a mask with compile-time bounds into padding and
  // rejects any other (a runtime bound, a compare that isn't offs < N).
  target.addLegalOp<arith::CmpIOp, arith::AndIOp>();
}

// Legal only when their condition holds (Conditions.cpp).
void addConditionallyLegal(ConversionTarget &target) {
  auto holds = [](Operation *op) {
    return succeeded(checkConditions(op, nullptr));
  };
  // unverified: tt.fp_to_fp's rounding mode isn't checked.
  target.addDynamicallyLegalOp<FuncOp, LoadOp, StoreOp, FpToFpOp, arith::ExtFOp,
                               arith::TruncFOp, arith::DivFOp, ReduceOp,
                               TransOp, ub::PoisonOp, scf::ForOp, DotOp>(holds);
}

// Atlas has an equivalent, but no pass lowers it yet.
void addNoLoweringYet(Rejection r) {
  r.reject<MakeTensorDescOp, DescriptorLoadOp, DescriptorStoreOp>(
      "descriptor kernels; Triton's descriptor rewrite leaves masks, i64 "
      "offsets and reshapes, not plain pointers");
  r.reject<DotScaledOp, ReshapeOp, CatOp, JoinOp, SplitOp, ScanOp, ScanReturnOp,
           MapElementwiseOp, MapElementwiseReturnOp>(
      "no Atlas pattern for this op");
  r.reject<UnsplatOp>("scalar data has no Atlas path in virtual IR");
  r.reject<ExternElementwiseOp>(
      "external library functions have no Atlas pattern");
  r.reject<PreciseSqrtOp, math::SqrtOp, math::RsqrtOp>(
      "Atlas's square root table has no pattern (rsqrt later as 1 / sqrt)");
  r.reject<math::TanhOp, math::SinOp, math::CosOp, math::Log2Op, math::LogOp>(
      "Atlas's table for this function is untested on hardware");
  r.reject<scf::WhileOp, scf::ConditionOp, scf::IfOp, scf::IndexSwitchOp,
           scf::ExecuteRegionOp, cf::BranchOp, cf::CondBranchOp, cf::SwitchOp>(
      "runtime control flow");
}

// No single Atlas op does this. Some could be built from other ops; none is
// supported, and some may never be.
void addNoDirectEquivalent(Rejection r) {
  r.reject<AtomicCASOp, AtomicLoadOp, AtomicPollOp, AtomicRMWOp, AtomicStoreOp>(
      "no atomic or read-modify-write memory instruction "
      "(could be built from a load, the op and a store on "
      "one core)");
  r.reject<GatherOp, DescriptorGatherOp, DescriptorScatterOp,
           DescriptorReduceOp, HistogramOp>(
      "data-dependent addresses; atlas-mlir accepts only constant whole-tile "
      "transfers");
  r.reject<ClampFOp, math::ClampFOp>(
      "no clamp op (could be built from a max then a min)");
  r.reject<ApproxDivFOp, PreciseDivFOp>(
      "no divide op (could be built from 1 / y through the reciprocal table, "
      "then a multiply)");
  r.reject<MulhiUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
           arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
           arith::AddUIExtendedOp, arith::SubUIExtendedOp,
           arith::MulSIExtendedOp, arith::MulUIExtendedOp, arith::TruncIOp,
           arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp,
           arith::IndexCastUIOp>(
      "an integer operation (address math uses addi and muli)");
  r.reject<arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
           arith::CeilDivUIOp, arith::FloorDivSIOp, arith::RemSIOp,
           arith::RemUIOp>("integer division and remainder");
  r.reject<arith::CmpFOp, arith::SelectOp, arith::OrIOp, arith::XOrIOp>(
      "float compares, selects, and boolean or and xor");
  r.reject<arith::NegFOp, arith::RemFOp>(
      "no op for this (negf could be built as x * -1)");
  r.reject<arith::BitcastOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
           arith::FPToUIOp, arith::ConvertFOp, arith::ScalingExtFOp,
           arith::ScalingTruncFOp, arith::FlushDenormalsOp>(
      "a conversion between integer and float, to other float formats, "
      "scaled, or flushing denormals");
  r.reject<math::AbsFOp, math::AbsIOp, math::AcosOp, math::AcoshOp,
           math::AsinOp, math::AsinhOp, math::AtanOp, math::Atan2Op,
           math::AtanhOp, math::CbrtOp, math::CeilOp, math::CopySignOp,
           math::CoshOp, math::CountLeadingZerosOp, math::CtPopOp,
           math::CountTrailingZerosOp, math::ErfOp, math::ErfcOp, math::ExpM1Op,
           math::FloorOp, math::FmaOp, math::FPowIOp, math::IPowIOp,
           math::IsFiniteOp, math::IsInfOp, math::IsNaNOp, math::IsNormalOp,
           math::Log10Op, math::Log1pOp, math::PowFOp, math::RoundOp,
           math::RoundEvenOp, math::SincosOp, math::SinhOp, math::TanOp,
           math::TruncOp>("no op for this function");
}

// Nothing to lower to on Atlas.
void addNoAtlasMeaning(Rejection r) {
  r.reject<ElementwiseInlineAsmOp, PrintOp, AssertOp, CallOp, IntToPtrOp,
           PtrToIntOp, BitcastOp, GridDependencyLaunchDependentsOp,
           GridDependencyWaitOp>("inline assembly, console, calls after "
                                 "inlining, pointer casts, grid dependencies");
  r.reject<scf::ForallOp, scf::InParallelOp, scf::ParallelOp, scf::ReduceOp,
           scf::ReduceReturnOp>(
      "parallel loop forms; Triton's front end doesn't emit them");
  r.reject<cf::AssertOp, ub::UnreachableOp>("an assertion or unreachable");
}

// Dropped: no meaning on one Atlas core, and harmless to erase (a barrier has
// nothing to wait for; an assume is only a hint). Illegal, so the conversion
// runs EraseDropped on them.
void addDropped(ConversionTarget &target) {
  target.addIllegalOp<gpu::BarrierOp, LLVM::AssumeOp>();
}

template <typename OpT> struct EraseDropped : OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void buildOpInventory(ConversionTarget &target, Reasons &reasons) {
  addLegal(target);
  addConditionallyLegal(target);
  addNoLoweringYet({target, reasons, "no lowering yet"});
  addNoDirectEquivalent({target, reasons, "no direct Atlas equivalent"});
  addNoAtlasMeaning({target, reasons, "no Atlas meaning"});
  addDropped(target);
  // An op outside the inventory isn't TTIR, so it is illegal too.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return false; });
}

void populateDropPatterns(RewritePatternSet &patterns) {
  patterns.add<EraseDropped<gpu::BarrierOp>, EraseDropped<LLVM::AssumeOp>>(
      patterns.getContext());
}

} // namespace mlir::triton::validate
