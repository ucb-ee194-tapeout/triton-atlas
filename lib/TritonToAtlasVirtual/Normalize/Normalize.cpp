// Pass 2, ttatlas-normalize. Contract: triton-atlas design agenda, page
// "Pass 2 · ttatlas-normalize". It moves a dot's bias out to a separate add,
// so every dot starts from zero or from the previous dot of its chain.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_NORMALIZE
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

// The dot that accumulates onto `dot` next: its only user, in the same block,
// taking `dot` as the accumulator (operand c). Null if there is none. A loop
// or branch boundary ends a chain.
DotOp nextInChain(DotOp dot) {
  if (!dot->hasOneUse())
    return nullptr;
  OpOperand &use = *dot->getUses().begin();
  auto next = dyn_cast<DotOp>(use.getOwner());
  return next && use.getOperandNumber() == 2 &&
                 next->getBlock() == dot->getBlock()
             ? next
             : nullptr;
}

// Whether `dot` starts a chain: its accumulator isn't a dot that continues
// into it.
bool startsChain(DotOp dot) {
  auto previous = dot.getC().getDefiningOp<DotOp>();
  return !previous || nextInChain(previous) != dot;
}

// The dots of every chain, in order: each chain is a run of dots where each
// one's only user is the next, as its accumulator (a K contraction).
SmallVector<SmallVector<DotOp>> findChains(ModuleOp module) {
  SmallVector<SmallVector<DotOp>> chains;
  module.walk([&](DotOp dot) {
    if (!startsChain(dot))
      return;
    SmallVector<DotOp> &chain = chains.emplace_back();
    for (DotOp step = dot; step; step = nextInChain(step))
      chain.push_back(step);
  });
  return chains;
}

// Atlas starts each matmul from zero: a bias on the first dot of a chain
// moves to one add after its last dot. Seeding the accumulator with a bias
// (VMATPUSH.ABF16) is listed as Untested in atlas-mlir's
// docs/hardware-status.md:52 @ 8ee175c, while matmul then a VPU add is the
// tested path (atlas-hw: pass 2 page, "Why unfold the bias"). Moving it past
// the whole chain, not just off the first dot, keeps the chain intact when a
// bias seeds a K loop: unrolled, the bias lands on dot 0, and an add there
// would split the chain. It changes the order of the float adds.
void unfoldBias(ArrayRef<DotOp> chain) {
  DotOp first = chain.front(), last = chain.back();
  Value bias = first.getC();
  auto type = cast<RankedTensorType>(bias.getType());
  // Integer dots don't pass pass 1 (it takes FP8 and BF16 loads only).
  if (!isa<FloatType>(type.getElementType()) ||
      matchPattern(bias, m_AnyZeroFloat()))
    return;
  OpBuilder builder(first);
  Value zero = arith::ConstantOp::create(builder, first.getLoc(), type,
                                         builder.getZeroAttr(type));
  first.getCMutable().assign(zero);
  builder.setInsertionPointAfter(last);
  auto add = arith::AddFOp::create(builder, last.getLoc(), last, bias);
  last.getResult().replaceAllUsesExcept(add, add);
}

struct Normalize : impl::NormalizeBase<Normalize> {
  void runOnOperation() override {
    for (SmallVector<DotOp> &chain : findChains(getOperation()))
      unfoldBias(chain);
  }
};

} // namespace

} // namespace mlir::triton
