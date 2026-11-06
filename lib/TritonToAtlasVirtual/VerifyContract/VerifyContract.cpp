// ttatlas-verify-contract: checks that IR meets the output contract of pass
// `stage`. Not in the pipeline; tests run it after the pass they test, and
// lanes run it on hand-written fixtures. Contracts: triton-atlas design agenda,
// each pass's page. Pass shape (one file, GEN_PASS_DEF, impl::...Base) follows
// triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_VERIFYCONTRACT
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;

bool isFp8(Type type) {
  return isa<Float8E4M3FNType>(getElementTypeOrSelf(type));
}

// The float element a value holds or points to (tensor elements, pointer
// pointees, nested), or null.
FloatType floatElement(Type type) {
  type = getElementTypeOrSelf(type);
  if (auto pointer = dyn_cast<PointerType>(type))
    return floatElement(pointer.getPointeeType());
  return dyn_cast<FloatType>(type);
}

// A tt.dot accumulator that pass 2 allows to start a contraction: a zero
// constant.
bool isZero(Value value) { return matchPattern(value, m_AnyZeroFloat()); }

struct VerifyContract : impl::VerifyContractBase<VerifyContract> {
  using Base::Base;

  void runOnOperation() override {
    failed = false;
    current = stage;
    ModuleOp module = getOperation();
    switch (current) {
    case 1:
      checkStage1(module);
      break;
    case 2:
      checkStage2(module);
      break;
    case 3:
      checkStage3(module);
      break;
    case 4:
      checkStage4(module);
      break;
    case 5:
      checkStage5(module);
      break;
    case 6:
      checkStage6(module);
      break;
    default:
      module.emitError() << "ttatlas-verify-contract needs stage=1..6, got "
                         << current;
      failed = true;
    }
    if (failed)
      signalPassFailure();
  }

  // Reports one broken promise.
  void broken(Operation *op, const Twine &what) {
    op->emitError() << "contract of pass " << current << " broken: " << what;
    failed = true;
  }

  // Every value an op defines or receives as a block argument, for checks on
  // all values.
  template <typename Fn> void forEachValue(Operation *op, Fn fn) {
    for (Value result : op->getResults())
      fn(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          fn(argument);
  }

  // Pass 1: one tt.func; host dtypes on every load and store.
  void checkStage1(ModuleOp module) {
    size_t funcs = llvm::range_size(module.getOps<FuncOp>());
    if (funcs != 1)
      broken(module, "expected one tt.func, found " + Twine(funcs));
    module.walk([&](Operation *op) {
      if (isa<LoadOp, StoreOp>(op) &&
          !op->getAttrOfType<TypeAttr>(tta::kHostDtype))
        broken(op, "missing ttatlas.host_dtype");
    });
  }

  // Pass 2: a dot starts with a zero accumulator, except a later step of a K
  // chain, which takes the previous step's result (same id and len, pos - 1);
  // no constant-trip scf.for.
  void checkStage2(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (auto dot = dyn_cast<DotOp>(op)) {
        auto chain = dot->getAttrOfType<tta::KChainAttr>(tta::kKChain);
        if (chain && chain.getPos() > 0) {
          auto producer = dot.getC().getDefiningOp<DotOp>();
          auto previous =
              producer ? producer->getAttrOfType<tta::KChainAttr>(tta::kKChain)
                       : nullptr;
          if (!previous || previous.getId() != chain.getId() ||
              previous.getLen() != chain.getLen() ||
              previous.getPos() != chain.getPos() - 1)
            broken(op, "ttatlas.k_chain step does not take the previous "
                       "step's result");
        } else if (!isZero(dot.getC())) {
          broken(op, "tt.dot accumulator is neither zero nor the previous step "
                     "of its ttatlas.k_chain");
        }
      }
      if (auto loop = dyn_cast<scf::ForOp>(op))
        if (getConstantIntValue(loop.getLowerBound()) &&
            getConstantIntValue(loop.getUpperBound()) &&
            getConstantIntValue(loop.getStep()))
          broken(op, "scf.for with a constant trip count is left");
    });
  }

  // Pass 3: every 2-D value, results and block arguments alike, is 32x32; an
  // MXU unit on every dot.
  void checkStage3(ModuleOp module) {
    module.walk([&](Operation *op) {
      forEachValue(op, [&](Value value) {
        if (auto tensor = dyn_cast<RankedTensorType>(value.getType());
            tensor && tensor.getRank() == 2 &&
            tensor.getShape() != ArrayRef<int64_t>{32, 32})
          broken(op, "2-D value is not 32x32");
      });
      if (isa<DotOp>(op) && !op->hasAttr(tta::kMxu))
        broken(op, "tt.dot without ttatlas.mxu");
    });
  }

  // Pass 4: FP8 dot operands; every other float value or pointer is BF16 or
  // FP8. K-chain accumulators stay in the MXU, so a chained dot's result and
  // the zero constant that starts a chain may be f32.
  void checkStage4(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (auto dot = dyn_cast<DotOp>(op))
        if (!isFp8(dot.getA().getType()) || !isFp8(dot.getB().getType()))
          broken(op, "tt.dot operands are not FP8 E4M3");
      bool chainDot = isa<DotOp>(op) && op->hasAttr(tta::kKChain);
      bool chainSeed = isa<arith::ConstantOp>(op) && !op->use_empty() &&
                       llvm::all_of(op->getUsers(), [](Operation *user) {
                         return isa<DotOp>(user) && user->hasAttr(tta::kKChain);
                       });
      if (chainDot || chainSeed)
        return;
      forEachValue(op, [&](Value value) {
        FloatType element = floatElement(value.getType());
        if (element && !element.isBF16() && !isFp8(element))
          broken(op, "float value or pointer is neither BF16 nor FP8");
      });
    });
  }

  // Pass 5: a slot on every load and store.
  void checkStage5(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (isa<LoadOp, StoreOp>(op) &&
          !op->getAttrOfType<tta::SlotAttr>(tta::kSlot))
        broken(op, "missing #ttatlas.slot");
    });
  }

  // Pass 6: no TTIR; one argument-free top-level func.func with the kernel
  // metadata, and no nested ones.
  void checkStage6(ModuleOp module) {
    size_t funcs = llvm::range_size(module.getOps<func::FuncOp>());
    if (funcs != 1)
      broken(module, "expected one func.func, found " + Twine(funcs));
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "tt" ||
          isa<UnrealizedConversionCastOp>(op))
        broken(op, Twine("TTIR is left: ") + op->getName().getStringRef());
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func)
        return;
      if (op->getParentOp() != module.getOperation())
        broken(op, "func.func nested inside another op");
      if (func.getNumArguments() != 0)
        broken(op, "func.func has arguments");
      if (!func->getAttrOfType<StringAttr>(tta::kKernel))
        broken(op, "func.func without ttatlas.kernel");
    });
  }

  unsigned current = 0;
  bool failed = false;
};

} // namespace

} // namespace mlir::triton
