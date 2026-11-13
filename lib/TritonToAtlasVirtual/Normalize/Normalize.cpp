// Pass 2, ttatlas-normalize. Contract: triton-atlas design agenda, page
// "Pass 2 · ttatlas-normalize". It unrolls every constant-trip scf.for, moves
// a dot's bias out to a separate add, so every dot starts from zero or from
// the previous dot of its chain, and tags each chain's dots ttatlas.k_chain.
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_NORMALIZE
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;

// Unrolls every scf.for, innermost first, with MLIR's loopUnrollFull (as the
// page says: no hand-written unroller). Pass 1 lets through only loops with a
// constant trip count. Why unroll: each K step loads a different tile, and
// atlas-mlir at our pin accepts only DMA addresses built from constants
// (lib/AtlasOps.cpp:196-201 @ 8ee175c), so a loop body can't name "the next
// tile" (atlas-hw: pass 2 page, "Why unroll constant-count loops"). Inner
// loops go first, so a bound computed from an outer loop's counter fails here;
// pass 1 rejects it as a runtime trip count.
// A loop that runs zero times (constant lower bound >= upper bound) is
// replaced by its initial values: loopUnrollFull succeeds on it without
// removing it, which would leave an scf.for behind.
LogicalResult unrollLoops(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) { loops.push_back(loop); }); // inner first
  IRRewriter rewriter(module.getContext());
  for (scf::ForOp loop : loops) {
    std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
    // Signed comparison, as scf.for does unless it is marked unsignedCmp;
    // an unsigned loop is left to loopUnrollFull.
    if (lower && upper && !loop.getUnsignedCmp() && *lower >= *upper) {
      rewriter.replaceOp(loop, loop.getInitArgs());
      continue;
    }
    if (failed(loopUnrollFull(loop)))
      return loop.emitError(
          "ttatlas-normalize: can't unroll this loop: its trip count isn't a "
          "constant (ttatlas-validate-ttir rejects such loops)");
  }
  return success();
}

// Program words one K step adds once unrolled: 23.5, measured by lowering
// with atlas-mlir 9e204d4 and running on npu_model 74982e7, for one 32x32
// output tile (mxu_accumulate per step, plus its loads); rounded down so the
// estimate stays a lower bound.
constexpr int64_t kWordsPerKStep = 23;

// Warns, without failing, when the unrolled kernel likely exceeds Atlas's
// program size: its dots alone, at kWordsPerKStep words each, already need
// more than kProgramWords (2048 at our pin). Only a warning, because this is
// an estimate before tiling (pass 4 multiplies the dots further) and the real
// size is checked at the end (pass 8); it tells the author early which loop
// made the program large.
void warnIfProgramTooLarge(ModuleOp module) {
  module.walk([](FuncOp func) {
    int64_t dots = 0;
    func.walk([&](DotOp) { ++dots; });
    int64_t words = dots * kWordsPerKStep;
    if (words > tta::target::kProgramWords)
      func.emitWarning() << "ttatlas-normalize: unrolled to " << dots
                         << " dots, at least " << words
                         << " program words; Atlas runs at most "
                         << tta::target::kProgramWords
                         << " words per program, so this kernel will likely "
                            "fail the program-size check";
  });
}

// The dot that accumulates onto `dot` next: its only user, in the same block,
// taking `dot` as the accumulator (operand c). Null if there is none. A loop
// or branch boundary ends a chain.
// Known limit: a dot whose result is also used elsewhere (a partial sum that
// is returned or stored) ends its chain, and the next dot starts a new chain.
// The result stays correct: the bias unfold moves the partial sum to an add
// after the new chain. It costs one extra MXU readout and VPU add where the
// MXU could have kept accumulating.
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

// Tags the dots of a chain of two or more with their place in it. Tagged by
// structure, not by loop index, so chains with no loop left also get tagged
// and pass the stage-2 check: those Triton already unrolled
// (tl.range(..., loop_unroll_factor=2)) and combine's dot-on-dot folds. A lone
// dot needs no tag: it starts from zero.
// Who reads the tag (TTAtlas.h kKChain, "pass 2, read by pass 3"): the stage-2
// contract check (ttatlas-verify-contract) and pass 3, which links the chain
// into one linalg.fill -> linalg.matmul -> ... def-use chain; after pass 3 the
// chain is that structure, not the tag. Pass 2 finds chains by structure too,
// but the tag is the contract between passes 2 and 3, so keep both.
void tagChain(MutableArrayRef<DotOp> chain, int32_t id) {
  int32_t len = chain.size();
  for (auto [pos, dot] : llvm::enumerate(chain))
    dot->setAttr(tta::kKChain,
                 tta::KChainAttr::get(dot.getContext(), id, int32_t(pos), len));
}

// Whether to keep the bias folded: the pass option or the module tag
// ttatlas.fold_bias_into_dot (set from Triton's AtlasOptions) turns it on;
// both default to off.
bool foldBias(ModuleOp module, bool option) {
  auto tag = module->getAttrOfType<BoolAttr>(tta::kFoldBiasIntoDot);
  return option || (tag && tag.getValue());
}

struct Normalize : impl::NormalizeBase<Normalize> {
  // Takes NormalizeOptions (fold-bias-into-dot).
  using NormalizeBase::NormalizeBase;

  void runOnOperation() override {
    // Unroll first: a K loop's dots form a chain only once unrolled, and the
    // bias unfold must see the whole chain.
    if (failed(unrollLoops(getOperation())))
      return signalPassFailure();
    warnIfProgramTooLarge(getOperation());
    bool fold = foldBias(getOperation(), foldBiasIntoDot);
    int32_t id = 0;
    for (SmallVector<DotOp> &chain : findChains(getOperation())) {
      // Folding keeps the bias as the first dot's accumulator: Atlas
      // then loads it into the MXU accumulator (VMATPUSH.ABF16, atlas-mlir's
      // virtual_mxu_load_acc_bf16) instead of adding it on the VPU after the
      // chain. Untested on hardware, so off by default (Normalize.td).
      if (!fold)
        unfoldBias(chain);
      if (chain.size() > 1)
        tagChain(chain, id++);
    }
  }
};

} // namespace

} // namespace mlir::triton
