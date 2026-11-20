// ttatlas-verify-contract: checks that IR meets the output contract of pass
// `stage`. Not in the pipeline; tests run it after the pass they test, and
// lanes run it on hand-written fixtures. Pass shape (one file, GEN_PASS_DEF,
// impl::...Base) follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/LaneValues.h"
#include "TritonToAtlasVirtual/PackTiles.h"
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// The float element a value holds, or null. A pointer holds none: loads and
// stores keep the host's pointer types.
FloatType floatElement(Type type) {
  return dyn_cast<FloatType>(getElementTypeOrSelf(type));
}

// The cast pass 5 puts between BF16 compute and a store to a host pointer type
// other than BF16: an extf or truncf read only as a store's value (CSE may
// merge the casts of several stores into one cast with several users).
bool isStoreCast(Operation *op) {
  if (!isa<arith::ExtFOp, arith::TruncFOp>(op))
    return false;
  return llvm::all_of(op->getUsers(), [&](Operation *user) {
    auto store = dyn_cast<StoreOp>(user);
    return store && store.getValue() == op->getResult(0);
  });
}

// A tt.dot accumulator that pass 2 allows to start a contraction: a zero
// constant.
bool isZero(Value value) { return matchPattern(value, m_AnyZeroFloat()); }

// linalg ops are matched by name: the plugin links only libtriton, which does
// not export linalg (the pass that creates linalg ops links it later).
bool isLinalg(Operation *op) {
  return op->getName().getDialectNamespace() == "linalg";
}
bool isMatmul(Operation *op) {
  return op->getName().getStringRef() == "linalg.matmul";
}
bool isFill(Operation *op) {
  return op->getName().getStringRef() == "linalg.fill";
}
// Inside a linalg op's body: scalar code that the op's own checks cover.
bool inLinalgBody(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isLinalg(parent))
      return true;
  return false;
}

namespace target = mlir::ttatlas::target;

// One Atlas tile: 32x32, or 32 long (a bias row, a row reduction's result).
bool isTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && (tensor.getRank() == 1 || tensor.getRank() == 2) &&
         llvm::all_of(tensor.getShape(),
                      [](int64_t size) { return size == target::kTileSize; });
}
bool isPack(Operation *op) {
  return op->getName().getStringRef() == "linalg.pack";
}
bool isUnpack(Operation *op) {
  return op->getName().getStringRef() == "linalg.unpack";
}
// The float casts that stand between a memory op and its pack or unpack.
bool isFloatCast(Operation *op) {
  return isa<arith::ExtFOp, arith::TruncFOp>(op);
}
// The memory op behind `value`, looking through float casts.
Operation *sourceBehindCasts(Value value) {
  Operation *op = value.getDefiningOp();
  while (op && isFloatCast(op))
    op = op->getOperand(0).getDefiningOp();
  return op;
}
// Every user of `value` is `kind` of op, looking through float casts.
bool onlyReadByStore(Value value) {
  return !value.use_empty() &&
         llvm::all_of(value.getUsers(), [](Operation *user) {
           return isa<StoreOp>(user) ||
                  (isFloatCast(user) && onlyReadByStore(user->getResult(0)));
         });
}

bool isFloatTensor(Type type) {
  return isa<RankedTensorType>(type) &&
         isa<FloatType>(getElementTypeOrSelf(type));
}

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
    case 7:
      checkStage7(module);
      break;
    default:
      module.emitError() << "ttatlas-verify-contract needs stage=1..7, got "
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
  // all values. A linalg op's body holds scalars of its operands' element
  // types, so only its results count.
  template <typename Fn> void forEachValue(Operation *op, Fn fn) {
    for (Value result : op->getResults())
      fn(result);
    if (isLinalg(op))
      return;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          fn(argument);
  }

  // Pass 1: one tt.func.
  void checkStage1(ModuleOp module) {
    size_t funcs = llvm::range_size(module.getOps<FuncOp>());
    if (funcs != 1)
      broken(module, "expected one tt.func, found " + Twine(funcs));
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

  // Pass 3: compute on float tensors is linalg; only constants, loads, stores
  // and address math (integer and pointer tensors) stay TTIR.
  void checkStage3(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (inLinalgBody(op))
        return;
      // Only float compute must be gone: pass 3 converts float tensors only,
      // and an integer transpose, dot or reduce (address math, or rejected by
      // pass 1) stays TTIR.
      // A reduce counts by its operands, as pass 3's target judges it (its
      // result can be a scalar).
      bool leftCompute =
          isa<ReduceOp>(op)
              ? llvm::any_of(op->getOperandTypes(), isFloatTensor)
              : isa<DotOp, TransOp>(op) &&
                    llvm::any_of(op->getResultTypes(), isFloatTensor);
      if (leftCompute) {
        broken(op,
               Twine("TTIR compute is left: ") + op->getName().getStringRef());
        return;
      }
      if (op->hasTrait<OpTrait::ConstantLike>())
        return;
      StringRef dialect = op->getName().getDialectNamespace();
      // arith / math on float tensors, whatever the result (cmpf gives i1);
      // any other tt op that makes a float tensor, except the load itself.
      bool floatCompute =
          (dialect == "arith" || dialect == "math")
              ? llvm::any_of(llvm::concat<const Type>(op->getOperandTypes(),
                                                      op->getResultTypes()),
                             isFloatTensor)
              : dialect == "tt" && !isa<LoadOp>(op) &&
                    llvm::any_of(op->getResultTypes(), isFloatTensor);
      if (floatCompute)
        broken(op,
               Twine("TTIR compute is left: ") + op->getName().getStringRef());
    });
  }

  // Pass 4: compute is linalg on single tiles (32x32, or 32 long); the memory
  // boundary is one linalg.pack of each tt.load (through float casts) and one
  // linalg.unpack of each stored value, so no pack reads an empty tensor, a
  // constant or an unpack; no tile loop is left and no tag of the old pass 4.
  // A kernel that only copies keeps its tt.load feeding its tt.store.
  void checkStage4(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (inLinalgBody(op))
        return;
      if (isa<scf::ForOp>(op))
        broken(op, "tile loop left");
      if (isPack(op)) {
        Operation *source = sourceBehindCasts(op->getOperand(0));
        if (!isa_and_nonnull<LoadOp>(source))
          broken(op, "linalg.pack of something other than a tt.load");
      } else if (isUnpack(op)) {
        if (!onlyReadByStore(op->getResult(0)))
          broken(op, "linalg.unpack read by something other than a tt.store");
      } else if (isLinalg(op)) {
        for (Type type : llvm::concat<const Type>(op->getOperandTypes(),
                                                  op->getResultTypes()))
          if (isa<RankedTensorType>(type) && !isTile(type)) {
            broken(op, "linalg value is not one tile (32 or 32x32)");
            break;
          }
      }
      if (auto load = dyn_cast<LoadOp>(op);
          load && isa<RankedTensorType>(load.getType())) {
        bool packed = llvm::all_of(load->getUsers(), [](Operation *user) {
          return isPack(user) || isFloatCast(user) || isa<StoreOp>(user);
        });
        if (!packed)
          broken(op, "tt.load read by something other than a linalg.pack");
      }
    });
  }

  // Pass 5: FP8 matmul inputs; every other float value is BF16 or FP8, inside
  // linalg bodies too, except the casts to a store's host type. Matmul
  // accumulators are BF16 like everything else.
  void checkStage5(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (isMatmul(op) &&
          !llvm::all_of(op->getOperands().take_front(2),
                        [](Value input) { return isFp8(input.getType()); }))
        broken(op, "linalg.matmul inputs are not FP8 E4M3");
      if (isStoreCast(op))
        return;
      auto check = [&](Value value) {
        FloatType element = floatElement(value.getType());
        if (element && !element.isBF16() && !isFp8(element))
          broken(op, "float value is neither BF16 nor FP8");
      };
      for (Value result : op->getResults())
        check(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments()) {
            // An unused scalar float kernel argument is the host's value and
            // keeps its type (pass 1 accepts one only unused).
            if (isa<FuncOp>(op) && isa<FloatType>(argument.getType()) &&
                argument.use_empty())
              continue;
            check(argument);
          }
    });
  }

  // Pass 6: every memory tile carries a slot: each linalg.pack and
  // linalg.unpack (one slot per tile), a tt.load or tt.store that no pack or
  // unpack stands behind (a copy), and each non-splat constant read by a
  // generic (constant slots).
  void checkStage6(ModuleOp module) {
    auto needSlot = [&](Operation *op) {
      Attribute slot = op->getAttr(tta::kSlot);
      if (!isa_and_nonnull<tta::SlotAttr, ArrayAttr>(slot))
        broken(op, "missing #ttatlas.slot");
    };
    module.walk([&](Operation *op) {
      if (auto pack = dyn_cast<linalg::PackOp>(op)) {
        needSlot(op);
      } else if (auto unpack = dyn_cast<linalg::UnPackOp>(op)) {
        needSlot(op);
      } else if (auto load = dyn_cast<LoadOp>(op)) {
        if (llvm::none_of(
                load->getUsers(),
                [](Operation *user) {
                  return isa<linalg::PackOp, arith::ExtFOp, arith::TruncFOp>(
                      user);
                }) &&
            !load->use_empty())
          needSlot(op);
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        if (!isa_and_nonnull<linalg::UnPackOp>(
                sourceBehindCasts(store.getValue())))
          needSlot(op);
      } else if (isLinalg(op) && !inLinalgBody(op) &&
                 op->getName().getStringRef() == "linalg.generic") {
        for (Value input : cast<linalg::LinalgOp>(op).getDpsInputs()) {
          auto slice = input.getDefiningOp<tensor::ExtractSliceOp>();
          Operation *source =
              slice ? slice.getSource().getDefiningOp() : input.getDefiningOp();
          auto constant = dyn_cast_or_null<arith::ConstantOp>(source);
          auto dense = constant
                           ? dyn_cast<DenseFPElementsAttr>(constant.getValue())
                           : DenseFPElementsAttr();
          if (!dense || dense.isSplat() ||
              !cast<RankedTensorType>(dense.getType())
                   .getElementType()
                   .isBF16())
            continue;
          needSlot(constant);
        }
      }
      if (isa<LoadOp, StoreOp>(op) && !isa<triton::FuncOp>(op->getParentOp()))
        broken(op, "nested boundary memory operation");
    });
  }

  // Pass 7: no TTIR; one argument-free top-level func.func with the kernel
  // metadata, and no nested ones.
  void checkStage7(ModuleOp module) {
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
