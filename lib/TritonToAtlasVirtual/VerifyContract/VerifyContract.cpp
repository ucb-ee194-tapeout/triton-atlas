// ttatlas-verify-contract: checks that IR meets the output contract of pass
// `stage`. Not in the pipeline; tests run it after the pass they test, and
// lanes run it on hand-written fixtures. Contracts: triton-atlas design agenda,
// each pass's page. Pass shape (one file, GEN_PASS_DEF, impl::...Base) follows
// triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
// A ranked 1-D or 2-D tensor that is not one tile.
bool isLarge(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && (tensor.getRank() == 1 || tensor.getRank() == 2) &&
         !isTile(tensor);
}
// A constant, unit-stride slice of one tile.
template <typename SliceOp> bool isTileSlice(SliceOp slice) {
  return llvm::all_of(slice.getStaticSizes(),
                      [](int64_t size) { return size == target::kTileSize; }) &&
         llvm::none_of(slice.getStaticOffsets(), ShapedType::isDynamic) &&
         llvm::all_of(slice.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}
// A tensor built only from constant tile insert_slices over a tensor.empty,
// one per tile: aligned, distinct, and covering the shape.
bool builtFromTiles(Value value) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(value.getType()).getShape();
  int64_t count = 1;
  for (int64_t size : shape) {
    if (size % target::kTileSize)
      return false;
    count *= size / target::kTileSize;
  }
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 16> tiles;
  while (auto insert = value.getDefiningOp<tensor::InsertSliceOp>()) {
    if (!isTileSlice(insert))
      return false;
    ArrayRef<int64_t> offset = insert.getStaticOffsets();
    for (auto [at, size] : llvm::zip(offset, shape))
      if (at % target::kTileSize || at >= size)
        return false;
    if (!tiles.insert({offset[0], offset.size() > 1 ? offset[1] : 0}).second)
      return false;
    value = insert.getDest();
  }
  return value.getDefiningOp<tensor::EmptyOp>() &&
         int64_t(tiles.size()) == count;
}

// A value read only as tiles: it is one tile, or every reader is a constant
// tile slice.
bool readAsTiles(Value value) {
  return isTile(value.getType()) ||
         llvm::all_of(value.getUsers(), [](Operation *user) {
           auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
           return slice && isTileSlice(slice);
         });
}

// The boundary form of a load smaller than, or not whole, tiles: its one
// reader is a tensor.pad that grows it to whole tiles (zero low padding) and
// whose result is read as tiles. Padded to one tile, the pad carries the
// ttatlas.pad tag; padded to several, each padded tile's slice does.
bool isPaddedBoundaryLoad(LoadOp load) {
  if (!load->hasOneUse())
    return false;
  auto pad = dyn_cast<tensor::PadOp>(*load->user_begin());
  if (!pad || pad.getSource() != load.getResult() ||
      !llvm::all_of(pad.getMixedLowPad(),
                    [](OpFoldResult low) { return isZeroInteger(low); }))
    return false;
  auto result = cast<RankedTensorType>(pad.getType());
  if (!llvm::all_of(result.getShape(),
                    [](int64_t size) { return size % target::kTileSize == 0; }))
    return false;
  if (isTile(result) && !pad->getAttrOfType<tta::PadAttr>(tta::kPad))
    return false;
  return readAsTiles(pad.getResult());
}

// The boundary form of a stored tensor smaller than, or not whole, tiles: the
// real lanes of a tensor that is one tile or built from tiles, cropped from
// the origin with unit strides. The tensor keeps every tile the crop touches.
bool isCroppedTiles(Value value) {
  auto crop = value.getDefiningOp<tensor::ExtractSliceOp>();
  if (!crop ||
      !llvm::all_of(crop.getMixedOffsets(),
                    [](OpFoldResult at) { return isZeroInteger(at); }) ||
      !llvm::all_of(crop.getMixedStrides(),
                    [](OpFoldResult stride) { return isOneInteger(stride); }))
    return false;
  auto source = cast<RankedTensorType>(crop.getSourceType());
  auto result = crop.getType();
  if (result.getRank() != source.getRank() || !result.hasStaticShape())
    return false;
  for (auto [kept, whole] : llvm::zip(result.getShape(), source.getShape()))
    if (llvm::divideCeil(kept, target::kTileSize) * target::kTileSize != whole)
      return false;
  return isTile(source) || builtFromTiles(crop.getSource());
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

  // Pass 4: linalg works on single tiles; an MXU unit on every matmul; every
  // fill writes a fresh tile-sized tensor.empty. A tt.load or tt.store larger
  // than a tile meets the tiles in one of these boundary forms:
  //  - a load read only through constant 32x32 slices;
  //  - a stored tensor built from one constant tile insert_slice per tile over
  //    a tensor.empty;
  //  - a load that is not whole tiles (a sub-tile load, or a 50-row one), read
  //    only by one tensor.pad up to whole tiles whose result is a tile or is
  //    read through tile slices: the host fills the padded DRAM slot, and the
  //    pad carries the ttatlas.pad tag when it makes one tile;
  //  - a stored value that is a constant slice from the origin (a crop) of a
  //    tile, or of a tensor built from tiles, that keeps every tile it touches.
  // Padded boundary tiles carry their own rank-2 ttatlas.pad tag (pass 4, see
  // LegalizeShapes.td); this check does not look at the tags.
  void checkStage4(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (inLinalgBody(op))
        return;
      if (isLinalg(op)) {
        for (Type type : llvm::concat<const Type>(op->getOperandTypes(),
                                                  op->getResultTypes()))
          if (isLarge(type)) {
            broken(op, "linalg value is not one tile (32 or 32x32)");
            break;
          }
        if (isMatmul(op) && !op->hasAttr(tta::kMxu))
          broken(op, "linalg.matmul without ttatlas.mxu");
        if (isFill(op)) {
          Value dest = op->getOperands().back();
          if (!dest.getDefiningOp<tensor::EmptyOp>() || !isTile(dest.getType()))
            broken(op, "linalg.fill does not write a fresh tensor.empty tile");
        }
      }
      if (auto load = dyn_cast<LoadOp>(op); load && isLarge(load.getType()) &&
                                            !readAsTiles(load.getResult()) &&
                                            !isPaddedBoundaryLoad(load))
        broken(op, "tt.load is read other than through constant tile slices");
      if (auto store = dyn_cast<StoreOp>(op);
          store && isLarge(store.getValue().getType()) &&
          !builtFromTiles(store.getValue()) &&
          !isCroppedTiles(store.getValue()))
        broken(op, "stored tensor is not built from constant tile "
                   "insert_slices");
    });
  }

  // Pass 5: FP8 matmul inputs; every other float value or pointer is BF16 or
  // FP8. A matmul chain's accumulator stays in the MXU, so the matmuls, the
  // linalg.fill that seeds them and what that fill reads and writes may be f32.
  void checkStage5(ModuleOp module) {
    llvm::SmallPtrSet<Operation *, 16> accumulator;
    module.walk([&](Operation *op) {
      if (!isMatmul(op))
        return;
      accumulator.insert(op);
      if (!llvm::all_of(op->getOperands().take_front(2),
                        [](Value input) { return isFp8(input.getType()); }))
        broken(op, "linalg.matmul inputs are not FP8 E4M3");
      Operation *seed = op->getOperands().back().getDefiningOp();
      if (seed && isFill(seed)) {
        accumulator.insert(seed);
        // The scalar zero and the tensor.empty the fill reads and writes,
        // when nothing else uses them.
        for (Value operand : seed->getOperands())
          if (Operation *def = operand.getDefiningOp();
              def && llvm::all_of(def->getUsers(), isFill))
            accumulator.insert(def);
      }
    });
    module.walk([&](Operation *op) {
      if (inLinalgBody(op) || accumulator.contains(op))
        return;
      forEachValue(op, [&](Value value) {
        FloatType element = floatElement(value.getType());
        if (element && !element.isBF16() && !isFp8(element))
          broken(op, "float value or pointer is neither BF16 nor FP8");
      });
    });
  }

  // Pass 6: a slot on every load and store (until it is settled whether the
  // tag moves to their slices).
  void checkStage6(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (isa<LoadOp, StoreOp>(op) &&
          !op->getAttrOfType<tta::SlotAttr>(tta::kSlot))
        broken(op, "missing #ttatlas.slot");
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
