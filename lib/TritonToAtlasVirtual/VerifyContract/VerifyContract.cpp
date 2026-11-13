// ttatlas-verify-contract: checks that IR meets the output contract of pass
// `stage`. Not in the pipeline; tests run it after the pass they test, and
// lanes run it on hand-written fixtures. Contracts: triton-atlas design agenda,
// each pass's page. Pass shape (one file, GEN_PASS_DEF, impl::...Base) follows
// triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/BoundaryTiles.h"
#include "TritonToAtlasVirtual/LaneValues.h"
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
// The readout marker pass 5 puts after a matmul chain: a linalg.generic with
// one input and one output whose body is one arith.truncf from f32 to BF16.
bool isReadout(Operation *op) {
  if (op->getName().getStringRef() != "linalg.generic" ||
      op->getNumOperands() != 2 || op->getNumRegions() != 1)
    return false;
  auto maps = op->getAttrOfType<ArrayAttr>("indexing_maps");
  if (!maps || !llvm::all_of(maps, [](Attribute map) {
        return cast<AffineMapAttr>(map).getValue().isIdentity();
      }))
    return false;
  Block &body = op->getRegion(0).front();
  auto narrow = dyn_cast<arith::TruncFOp>(&body.front());
  return body.getOperations().size() == 2 && narrow &&
         narrow.getIn() == body.getArgument(0) &&
         body.getTerminator()->getOperand(0) == narrow.getResult() &&
         narrow.getIn().getType().isF32() && narrow.getType().isBF16() &&
         getElementTypeOrSelf(op->getResult(0).getType()).isBF16();
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

// The boundary form of a load smaller than, or not whole, tiles: every reader
// is a tensor.pad that grows it to whole tiles (zero low padding) and whose
// result is read as tiles; the pads are one per reading op until CSE merges
// them, and they all carry the same fill, since the host fills a load once.
// Padded to one tile, the pad carries the ttatlas.pad tag; padded to several,
// each padded tile's slice does.
bool isPaddedBoundaryLoad(LoadOp load) {
  if (load->use_empty())
    return false;
  tta::PadAttr fill;
  for (Operation *user : load->getUsers()) {
    auto pad = dyn_cast<tensor::PadOp>(user);
    if (!pad || pad.getSource() != load.getResult() ||
        !llvm::all_of(pad.getMixedLowPad(),
                      [](OpFoldResult low) { return isZeroInteger(low); }))
      return false;
    auto result = cast<RankedTensorType>(pad.getType());
    if (!llvm::all_of(result.getShape(), [](int64_t size) {
          return size % target::kTileSize == 0;
        }))
      return false;
    auto tag = pad->getAttrOfType<tta::PadAttr>(tta::kPad);
    if (isTile(result) && !tag)
      return false;
    if (tag) {
      if (fill && fill.getNeutral() != tag.getNeutral())
        return false;
      fill = tag;
    }
    if (!readAsTiles(pad.getResult()))
      return false;
  }
  return true;
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

// The tiles of a stored value and where each sits, as [row, col]: the value
// itself when it is one tile, else the constant tile insert_slices that build
// it. Each tile is the op that yields it (null for a block argument). Empty
// when the value has neither form.
SmallVector<std::pair<Operation *, std::pair<int64_t, int64_t>>>
storedTiles(Value value) {
  SmallVector<std::pair<Operation *, std::pair<int64_t, int64_t>>> tiles;
  if (isTile(value.getType())) {
    tiles.push_back({value.getDefiningOp(), {0, 0}});
    return tiles;
  }
  if (!builtFromTiles(value))
    return tiles;
  while (auto insert = value.getDefiningOp<tensor::InsertSliceOp>()) {
    ArrayRef<int64_t> offset = insert.getStaticOffsets();
    tiles.push_back({insert.getSource().getDefiningOp(),
                     {offset.size() > 1 ? offset[0] : 0, offset.back()}});
    value = insert.getDest();
  }
  return tiles;
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
  //    only by tensor.pads up to whole tiles (one fill for all of them) whose
  //    result is a tile or is read through tile slices: the pad carries the
  //    ttatlas.pad tag when it makes one tile;
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

  // Pass 4 promised that the producer of every stored tile with padded lanes
  // carries a ttatlas.pad tag (the tile's logical shape, for pass 6). Pass 5
  // erases casts and pass-through generics, so a tag that did not move with
  // its value is lost here. The real lanes of the store are its crop from the
  // origin, cut down by its mask, and a tile is partial when they do not
  // fill it.
  void checkBoundaryTags(ModuleOp module) {
    module.walk([&](StoreOp store) {
      Value value = store.getValue();
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() > 2)
        return;
      SmallVector<int64_t> real(type.getShape());
      if (auto crop = value.getDefiningOp<tensor::ExtractSliceOp>();
          crop &&
          llvm::all_of(crop.getMixedOffsets(),
                       [](OpFoldResult at) { return isZeroInteger(at); }) &&
          crop.getType().hasStaticShape() &&
          crop.getType().getRank() == crop.getSourceType().getRank()) {
        real.assign(crop.getType().getShape().begin(),
                    crop.getType().getShape().end());
        value = crop.getSource();
      }
      if (Value mask = store.getMask())
        if (std::optional<SmallVector<int64_t>> bounds = maskBounds(mask);
            bounds && bounds->size() == real.size())
          for (auto [at, bound] : llvm::zip(real, *bounds))
            at = std::min(at, bound);
      for (auto [producer, offset] : storedTiles(value)) {
        SmallVector<int64_t> origin = {offset.first, offset.second};
        if (real.size() == 1)
          origin = {offset.second};
        bool partial = false, empty = false;
        for (auto [axis, at] : llvm::enumerate(real)) {
          int64_t inside = at - origin[axis];
          empty |= inside <= 0;
          partial |= inside < target::kTileSize;
        }
        // A rank-1 tile is a row of 32 lanes.
        if (partial && !empty && (!producer || !producer->hasAttr(tta::kPad)))
          broken(store, "a stored tile with padded lanes has no ttatlas.pad "
                        "on the op that produces it");
      }
    });
  }

  // Pass 5: FP8 matmul inputs; every other float value or pointer is BF16 or
  // FP8, inside linalg bodies too. A matmul chain's accumulator stays in the
  // MXU, so the matmuls, the linalg.fill that seeds them and what that fill
  // reads and writes may be f32; each chain's last matmul feeds exactly one
  // readout marker, a linalg.generic{arith.truncf f32 -> bf16}, whose body
  // reads that f32.
  void checkStage5(ModuleOp module) {
    checkBoundaryTags(module);
    llvm::SmallPtrSet<Operation *, 16> accumulator, readouts;
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
      // The chain's last matmul: no matmul accumulates onto it.
      bool continued = llvm::any_of(op->getUses(), [](OpOperand &use) {
        return isMatmul(use.getOwner()) &&
               use.getOperandNumber() == use.getOwner()->getNumOperands() - 1;
      });
      if (continued)
        return;
      if (op->hasOneUse() && isReadout(*op->user_begin()))
        readouts.insert(*op->user_begin());
      else
        broken(op, "the chain's last linalg.matmul doesn't feed exactly one "
                   "readout (linalg.generic{arith.truncf f32 -> bf16})");
    });
    module.walk([&](Operation *op) {
      if (accumulator.contains(op))
        return;
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp())
        if (accumulator.contains(parent) || readouts.contains(parent))
          return;
      auto check = [&](Value value) {
        FloatType element = floatElement(value.getType());
        if (element && !element.isBF16() && !isFp8(element))
          broken(op, "float value or pointer is neither BF16 nor FP8");
      };
      for (Value result : op->getResults())
        check(result);
      if (readouts.contains(op))
        return;
      // Only the chain and its readout may read the f32 accumulator; anything
      // else (a partial sum stored as is) would need f32 outside the MXU.
      // Other f32 values are flagged where they are defined.
      for (Value operand : op->getOperands())
        if (Operation *def = operand.getDefiningOp();
            def && accumulator.contains(def) && isMatmul(def))
          broken(op, "reads a matmul chain's f32 accumulator outside its "
                     "chain and readout");
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

  // Pass 6: every logical boundary tile carries a slot. Large memory
  // operations are represented by their constant extract/insert slices.
  void checkStage6(ModuleOp module) {
    for (auto function : module.getOps<triton::FuncOp>()) {
      SmallVector<BoundaryTile> tiles;
      llvm::SmallPtrSet<Operation *, 32> scaffolding;
      if (mlir::failed(collectBoundaryTiles(function, tiles, scaffolding))) {
        broken(function, "invalid boundary tile coverage");
        continue;
      }
      for (const auto &tile : tiles)
        if (!tile.tag->getAttrOfType<tta::SlotAttr>(tta::kSlot))
          broken(tile.tag, "missing #ttatlas.slot");
    }
    module.walk([&](Operation *op) {
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
