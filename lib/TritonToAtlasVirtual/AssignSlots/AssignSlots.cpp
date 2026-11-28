// Pass 6: one slot per memory tile. After pass 4 the memory boundary is a
// linalg.pack of each tt.load and a linalg.unpack of each stored value, and
// each tile of them is one op (PackTiles.h). This pass reads the pointer
// math, mask and `other` of the tt.load / tt.store with the shared lane
// evaluator and puts a #ttatlas.slot on every tile op; clamp constants get
// constant slots; a kernel with no compute gets its slot on the tt.load and
// tt.store themselves.
#include "TritonToAtlasVirtual/LaneValues.h"
#include "TritonToAtlasVirtual/PackTiles.h"
#include "TritonToAtlasVirtual/Passes.h"
#include "TritonToAtlasVirtual/VectorLayouts.h"

#include "TTAtlas/AtlasTarget.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iterator>

namespace mlir::triton {

#define GEN_PASS_DEF_ASSIGNSLOTS
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

using ttatlas::SlotAttr;
using ttatlas::SlotKind;
using ttatlas::SlotRole;
using ttatlas::VectorLayout;
constexpr int64_t kTile = ttatlas::target::kTileSize;

// What one tt.load or tt.store says about memory.
struct Access {
  Operation *op;
  BlockArgument arg;
  SmallVector<int64_t> shape;   // the block: [rows, cols] or [lanes]
  int64_t offset;               // flat element offset of lane 0
  int64_t rowStride, colStride; // a vector's lane stride is rowStride
  LaneBox valid;                // the mask's box (the block when unmasked)
  FloatAttr other;              // the masked lanes' value, if the load has one
  Type hostDtype;
  bool vector() const { return shape.size() == 1; }
};

// The splat value of a constant tensor (a dense splat, or tt.splat of a scalar
// constant), if it is one.
std::optional<APFloat> splatValue(Value value) {
  FloatAttr scalar;
  if (auto splat = value.getDefiningOp<SplatOp>())
    value = splat.getSrc();
  if (matchPattern(value, m_Constant(&scalar)))
    return scalar.getValue();
  DenseFPElementsAttr dense;
  if (matchPattern(value, m_Constant(&dense)) && dense.isSplat())
    return dense.getSplatValue<APFloat>();
  return std::nullopt;
}

bool dependsOnProgramId(Value value) {
  SetVector<Operation *> slice;
  (void)getBackwardSlice(value, &slice);
  return llvm::any_of(slice, [](Operation *op) {
    return isa<GetProgramIdOp, GetNumProgramsOp>(op);
  });
}

// The tt.load behind a value, looking through float casts.
Operation *sourceBehindCasts(Value value) {
  Operation *op = value.getDefiningOp();
  while (op && isa<arith::ExtFOp, arith::TruncFOp>(op))
    op = op->getOperand(0).getDefiningOp();
  return op;
}

struct Assigner {
  explicit Assigner(FuncOp func) : func(func) {}
  FuncOp func;
  int32_t nextInput = 0, nextOutput = 0;
  DenseMap<Attribute, int32_t> inputIndex; // slot with index 0 -> its index
  DenseMap<Operation *, Access> accesses;
  VectorLayouts layouts;
  SmallVector<linalg::PackOp> extraPacks; // clones made by splitByLayout

  bool reject(Operation *op, const Twine &reason) {
    ttatlas::emitUnsupported(op, reason);
    return false;
  }

  // ---- memory facts ------------------------------------------------------

  std::optional<Access> analyze(Operation *op) {
    auto load = dyn_cast<LoadOp>(op);
    auto store = dyn_cast<StoreOp>(op);
    Value ptr = load ? load.getPtr() : store.getPtr();
    Value mask = load ? load.getMask() : store.getMask();
    if (load && load.getIsVolatile())
      return reject(op, "volatile loads cannot be represented by an input "
                        "slot"),
             std::nullopt;
    auto type = dyn_cast<RankedTensorType>(ptr.getType());
    if (!type || type.getRank() < 1 || type.getRank() > 2 ||
        !type.hasStaticShape() || !isa<PointerType>(type.getElementType()))
      return reject(op, "expected a tensor of pointers of rank 1 or 2 "
                        "(tensor and block pointers are unsupported)"),
             std::nullopt;
    Access access{
        op, {}, SmallVector<int64_t>(type.getShape()),
        0,  0,  0,
        {}, {}, cast<PointerType>(type.getElementType()).getPointeeType()};
    int64_t lanes = type.getNumElements();
    if (lanes > kMaxEvaluatedLanes)
      return reject(op, "a block of more than 65536 lanes (256x256) is over "
                        "the evaluator's limit"),
             std::nullopt;
    std::optional<PointerLanes> pointer = evaluatePointer(ptr);
    if (!pointer)
      return reject(op, dependsOnProgramId(ptr)
                            ? "the pointer depends on a program id (a program "
                              "grid is not supported yet)"
                            : "cannot prove a pointer that is a kernel "
                              "argument plus compile-time offsets"),
             std::nullopt;
    access.arg = pointer->base;
    ArrayRef<int64_t> offsets = pointer->offsets.values;
    access.offset = offsets[0];
    if (access.vector()) {
      access.colStride = 1;
      access.rowStride = lanes > 1 ? offsets[1] - offsets[0] : 1;
    } else {
      int64_t rows = type.getDimSize(0), cols = type.getDimSize(1);
      access.colStride = cols > 1 ? offsets[1] - offsets[0] : 1;
      access.rowStride =
          rows > 1 ? offsets[cols] - offsets[0] : access.colStride * cols;
    }
    for (int64_t lane = 0; lane < lanes; ++lane) {
      int64_t row = access.vector() ? 0 : lane / access.shape[1];
      int64_t col = access.vector() ? lane : lane % access.shape[1];
      int64_t stride = access.vector() ? access.rowStride : access.colStride;
      if (offsets[lane] !=
          access.offset + row * access.rowStride + col * stride)
        return reject(op, "a non-constant stride: the tile is not an affine "
                          "gather"),
               std::nullopt;
    }
    if (access.rowStride < 0 || access.colStride < 0)
      return reject(op, "a negative stride: the packer reads tiles forward "
                        "from the argument"),
             std::nullopt;
    if (mask) {
      std::optional<LaneBox> box = maskBox(mask);
      if (!box)
        return reject(op, dependsOnProgramId(mask)
                              ? "the mask depends on a program id (a program "
                                "grid is not supported yet)"
                              : "a mask that is not one compile-time "
                                "rectangle"),
               std::nullopt;
      access.valid = *box;
    } else {
      access.valid.lo.assign(access.shape.size(), 0);
      access.valid.hi.assign(access.shape.begin(), access.shape.end());
    }
    if (load && mask && load.getOther()) {
      std::optional<APFloat> other = splatValue(load.getOther());
      auto element = dyn_cast<FloatType>(access.hostDtype);
      if (!other || !element)
        return reject(op, "a masked load's other must be a constant"),
               std::nullopt;
      if (isa<Float8E4M3FNType>(element) &&
          (other->isInfinity() || other->isNaN()))
        return reject(op, "an FP8 slot cannot hold a non-finite other"),
               std::nullopt;
      bool lost;
      APFloat converted = *other;
      converted.convert(element.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &lost);
      access.other = FloatAttr::get(element, converted);
    }
    return access;
  }

  // Memory dependences: stores must not overlap, and a load must not read what
  // an earlier store wrote. Footprints are the valid lanes' flat offsets.
  bool checkDependences(ArrayRef<Operation *> memoryOps) {
    DenseMap<unsigned, DenseSet<int64_t>> written;
    for (Operation *op : memoryOps) {
      const Access &access = accesses.at(op);
      auto ptr = isa<LoadOp>(op) ? cast<LoadOp>(op).getPtr()
                                 : cast<StoreOp>(op).getPtr();
      DenseSet<int64_t> footprint;
      int64_t lanes = cast<RankedTensorType>(ptr.getType()).getNumElements();
      int64_t cols = access.vector() ? lanes : access.shape[1];
      for (int64_t lane = 0; lane < lanes; ++lane) {
        int64_t row = access.vector() ? 0 : lane / cols,
                col = access.vector() ? lane : lane % cols;
        bool in = access.vector()
                      ? col >= access.valid.lo[0] && col < access.valid.hi[0]
                      : row >= access.valid.lo[0] && row < access.valid.hi[0] &&
                            col >= access.valid.lo[1] &&
                            col < access.valid.hi[1];
        if (!in)
          continue;
        int64_t stride = access.vector() ? access.rowStride : access.colStride;
        if (!footprint
                 .insert(access.offset + row * access.rowStride + col * stride)
                 .second &&
            isa<StoreOp>(op))
          return reject(op, "overlapping store footprints");
      }
      DenseSet<int64_t> &stored = written[access.arg.getArgNumber()];
      for (int64_t offset : footprint)
        if (stored.contains(offset))
          return reject(op, isa<StoreOp>(op)
                                ? "overlapping store footprints"
                                : "a load reads what an earlier store wrote");
      if (isa<StoreOp>(op))
        stored.insert(footprint.begin(), footprint.end());
    }
    return true;
  }

  // ---- slots -------------------------------------------------------------

  // The slot of tile (row, col) of a packed access. `tile` is the tile's own
  // dtype, `transposed` whether the pack turns the tile.
  SlotAttr memorySlot(Operation *at, const Access &access, int64_t row,
                      int64_t col, SlotRole role, VectorLayout layout,
                      bool transposed, Type tileDtype) {
    SmallVector<int64_t> block, valid;
    int64_t offset;
    auto clip = [&](int64_t lo, int64_t hi, int64_t origin, int64_t extent,
                    int64_t &outLo, int64_t &outHi) {
      outLo = std::max(lo, origin) - origin;
      outHi = std::min(hi, origin + extent) - origin;
      if (outHi <= outLo)
        outLo = outHi = 0;
    };
    if (access.vector()) {
      int64_t origin = row * kTile;
      int64_t extent = std::min(kTile, access.shape[0] - origin);
      int64_t lo, hi;
      clip(access.valid.lo[0], access.valid.hi[0], origin, extent, lo, hi);
      // A vector's lanes run along dim 0 of block and valid.
      block = {0, 0, extent, 1};
      valid = lo == hi ? SmallVector<int64_t>{0, 0, 0, 0}
                       : SmallVector<int64_t>{lo, 0, hi, 1};
      offset = access.offset + origin * access.rowStride;
    } else {
      int64_t r0 = row * kTile, c0 = col * kTile;
      int64_t rows = std::min(kTile, access.shape[0] - r0);
      int64_t cols = std::min(kTile, access.shape[1] - c0);
      int64_t rlo, rhi, clo, chi;
      clip(access.valid.lo[0], access.valid.hi[0], r0, rows, rlo, rhi);
      clip(access.valid.lo[1], access.valid.hi[1], c0, cols, clo, chi);
      block = {0, 0, rows, cols};
      valid = (rlo == rhi || clo == chi)
                  ? SmallVector<int64_t>{0, 0, 0, 0}
                  : SmallVector<int64_t>{rlo, clo, rhi, chi};
      offset = access.offset + r0 * access.rowStride + c0 * access.colStride;
    }
    if (offset < 0) {
      reject(at, "a tile whose first element is before the argument");
      return {};
    }
    auto emitError = [&]() { return at->emitError(); };
    return SlotAttr::getChecked(
        llvm::function_ref<InFlightDiagnostic()>(emitError), at->getContext(),
        int32_t(0), SlotKind::memory, int32_t(access.arg.getArgNumber()), role,
        offset, access.rowStride, access.colStride, ArrayRef<int64_t>(block),
        ArrayRef<int64_t>(valid), access.other, DenseIntElementsAttr(), layout,
        transposed, access.hostDtype, tileDtype);
  }

  // Numbers a slot: identical input slots (one tile read the same way twice)
  // share an index, an output is always new.
  SlotAttr number(SlotAttr slot, bool input) {
    auto withIndex = [&](int32_t index) {
      return SlotAttr::get(
          slot.getContext(), index, slot.getKind(), slot.getArg(),
          slot.getRole(), slot.getOffset(), slot.getRowStride(),
          slot.getColStride(), slot.getBlock(), slot.getValid(),
          slot.getOther(), slot.getPattern(), slot.getVectorLayout(),
          slot.getTransposed(), slot.getHostDtype(), slot.getTileDtype());
    };
    if (!input)
      return withIndex(nextOutput++);
    auto [it, inserted] = inputIndex.try_emplace(slot, nextInput);
    if (inserted)
      ++nextInput;
    return withIndex(it->second);
  }

  // ---- roles ---------------------------------------------------------------

  // What a tile that a pack yields is for, from the contraction that reads it
  // (operand 0 an activation, operand 1 a weight, host NK or KN from the maps
  // and the pack's transpose); any other FP8 reader is rejected.
  std::optional<SlotRole> inputRole(Operation *tileOp, bool vector,
                                    bool transposed, Type tileDtype) {
    std::optional<SlotRole> role;
    auto note = [&](SlotRole next) {
      if (role && *role != next)
        return false;
      role = next;
      return true;
    };
    for (OpOperand &use : tileOp->getResult(0).getUses()) {
      auto generic = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (!generic || !linalg::isaContractionOpInterface(generic))
        continue;
      if (use.getOperandNumber() == 0) {
        if (!note(SlotRole::activation_fp8))
          return std::nullopt;
        continue;
      }
      // The device tile of B is [n][k] or [k][n]; the host tile is that or its
      // transpose.
      auto dims = linalg::inferContractionDims(generic);
      if (failed(dims) || use.getOperandNumber() != 1)
        return std::nullopt;
      AffineMap map = generic.getMatchingIndexingMap(&use);
      auto isK = [&](AffineExpr expr) {
        auto dim = dyn_cast<AffineDimExpr>(expr);
        return dim && llvm::is_contained(dims->k, dim.getPosition());
      };
      bool deviceNK = isK(map.getResult(map.getNumResults() - 1));
      if (!note(deviceNK != transposed ? SlotRole::weight_fp8_nk
                                       : SlotRole::weight_fp8_kn))
        return std::nullopt;
    }
    if (role)
      return role;
    if (isa<Float8E4M3FNType>(tileDtype))
      return std::nullopt;
    return vector ? SlotRole::bias_bf16 : SlotRole::constant_bf16;
  }

  // ---- the passes' walks -----------------------------------------------

  // Sets the slots of a boundary op: one slot per tile in row-major grid
  // order, as an ArrayAttr, or the bare slot for a single tile.
  static void setSlots(Operation *op, ArrayRef<Attribute> slots) {
    if (slots.size() == 1)
      op->setAttr(ttatlas::kSlot, slots.front());
    else
      op->setAttr(ttatlas::kSlot, ArrayAttr::get(op->getContext(), slots));
  }

  // A loaded vector read as rows by some readers and as columns by others
  // needs two slots (the host writes it once per layout). The pack and the
  // tile ops of the readers of each extra layout are cloned; the readers move
  // to the clones, which are returned to be assigned in their turn.
  SmallVector<linalg::PackOp> splitByLayout(linalg::PackOp pack,
                                            ArrayRef<PackTile> tiles) {
    SmallVector<linalg::PackOp> clones;
    std::optional<VectorLayout> first;
    DenseMap<unsigned, linalg::PackOp> cloneOf;
    OpBuilder builder(pack);
    builder.setInsertionPointAfter(pack);
    for (PackTile tile : tiles) {
      DenseMap<unsigned, SmallVector<OpOperand *>> moved;
      for (OpOperand &use : tile.op->getResult(0).getUses()) {
        VectorLayout wanted = layouts.ofUse(use);
        first = first.value_or(wanted);
        if (wanted != *first)
          moved[unsigned(wanted)].push_back(&use);
      }
      for (auto &[layout, uses] : moved) {
        linalg::PackOp &copy = cloneOf[layout];
        if (!copy) {
          copy = cast<linalg::PackOp>(builder.clone(*pack));
          clones.push_back(copy);
        }
        IRMapping mapping;
        mapping.map(pack.getResult(), copy.getResult());
        OpBuilder after(tile.op->getContext());
        after.setInsertionPointAfter(tile.op);
        Operation *tileCopy = after.clone(*tile.op, mapping);
        for (OpOperand *use : uses)
          use->set(tileCopy->getResult(0));
      }
    }
    return clones;
  }

  bool assignInputs(linalg::PackOp pack) {
    auto load = dyn_cast_or_null<LoadOp>(sourceBehindCasts(pack.getSource()));
    if (!load) {
      // A plugin-runtime pipeline may retain a pack around a named linalg
      // result because named-op generalization is disabled for the wheel's
      // incompatible linalg trait table.  It is an internal tile reshape,
      // not a DRAM boundary, so it intentionally carries no input slots.
      if (std::getenv("TTATLAS_PLUGIN_RUNTIME"))
        return true;
      return reject(pack, "a pack of something other than a tt.load");
    }
    const Access &access = accesses.at(load);
    Type tileDtype = pack.getDestType().getElementType();
    if (!tileDtype.isBF16() && !isa<Float8E4M3FNType>(tileDtype))
      return reject(load, "expected BF16 or FP8 E4M3FN tiles");
    bool transposed =
        pack.getInnerDimsPos().size() == 2 && pack.getInnerDimsPos()[0] == 1;
    std::optional<SmallVector<PackTile>> tiles = inputTiles(pack);
    if (!tiles)
      return false;
    if (access.vector())
      llvm::append_range(extraPacks, splitByLayout(pack, *tiles));
    // The role comes from the readers of the tiles that are read; an unread
    // tile (the grid is padded to the readers' needs) takes the same role.
    std::optional<SlotRole> role;
    VectorLayout layout =
        access.vector() ? VectorLayout::rows : VectorLayout::none;
    bool layoutSet = false;
    for (PackTile tile : *tiles) {
      std::optional<SlotRole> tileRole =
          inputRole(tile.op, access.vector(), transposed, tileDtype);
      if (!tileRole || (role && *role != *tileRole))
        return reject(tile.op, "cannot infer a unique activation or weight "
                               "role from the contractions that read this "
                               "tile");
      role = tileRole;
      if (access.vector())
        for (OpOperand &use : tile.op->getResult(0).getUses()) {
          if (!layoutSet)
            layout = layouts.ofUse(use);
          layoutSet = true;
        }
    }
    if (!role)
      role = inputRole(pack, access.vector(), transposed, tileDtype);
    if (!role)
      return reject(pack, "cannot infer a role for this load");
    ArrayRef<int64_t> grid = pack.getDestType().getShape();
    int64_t rows = grid[0], cols = access.vector() ? 1 : grid[1];
    SmallVector<Attribute> slots;
    for (int64_t row = 0; row < rows; ++row)
      for (int64_t col = 0; col < cols; ++col) {
        SlotAttr slot = memorySlot(pack, access, row, col, *role, layout,
                                   transposed, tileDtype);
        if (!slot)
          return false;
        slots.push_back(number(slot, /*input=*/true));
      }
    setSlots(pack, slots);
    return true;
  }

  bool assignOutputs(linalg::UnPackOp unpack) {
    // The stores the unpack feeds, through float casts. One slot array
    // describes one place in memory, so a second store is rejected.
    SmallVector<StoreOp> stores;
    SmallVector<Operation *> work(unpack->user_begin(), unpack->user_end());
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      if (isa<arith::ExtFOp, arith::TruncFOp>(op))
        work.append(op->user_begin(), op->user_end());
      else if (auto s = dyn_cast<StoreOp>(op))
        stores.push_back(s);
    }
    if (stores.empty())
      return reject(unpack, "a linalg.unpack that no tt.store writes");
    if (stores.size() > 1)
      return reject(unpack, "a stored tile written to two places");
    StoreOp store = stores.front();
    const Access &access = accesses.at(store);
    Type tileDtype = unpack.getSourceType().getElementType();
    if (!tileDtype.isBF16())
      return reject(store, "FP8 stores have no Atlas virtual output "
                           "boundary operation");
    bool transposed = unpack.getInnerDimsPos().size() == 2 &&
                      unpack.getInnerDimsPos()[0] == 1;
    std::optional<SmallVector<PackTile>> tiles = outputTiles(unpack);
    if (!tiles)
      return false;
    SmallVector<Attribute> slots;
    for (PackTile tile : *tiles) {
      VectorLayout layout = VectorLayout::none;
      if (access.vector()) {
        Value value = isa<tensor::InsertSliceOp>(tile.op)
                          ? cast<tensor::InsertSliceOp>(tile.op).getSource()
                          : cast<tensor::ExpandShapeOp>(tile.op).getSrc();
        layout = layouts.of(value);
      }
      SlotAttr slot =
          memorySlot(unpack, access, tile.row, tile.col, SlotRole::output_bf16,
                     layout, transposed, tileDtype);
      if (!slot)
        return false;
      slots.push_back(number(slot, /*input=*/false));
    }
    setSlots(unpack, slots);
    return true;
  }

  // A kernel with no compute: the slots go on the tt.load / tt.store
  // themselves, one per tile of the block in row-major grid order.
  bool assignCopy(Operation *op, bool input) {
    const Access &access = accesses.at(op);
    Type tileDtype = access.hostDtype;
    if (!tileDtype.isBF16() && !isa<Float8E4M3FNType>(tileDtype))
      return reject(op, "a copy of a host type other than BF16 or FP8 "
                        "has no tile dtype");
    if (!input && !tileDtype.isBF16())
      return reject(op, "FP8 stores have no Atlas virtual output boundary "
                        "operation");
    SlotRole role = input ? (access.vector() ? SlotRole::bias_bf16
                                             : SlotRole::constant_bf16)
                          : SlotRole::output_bf16;
    int64_t rows = llvm::divideCeil(access.shape[0], kTile);
    int64_t cols =
        access.vector() ? 1 : llvm::divideCeil(access.shape[1], kTile);
    SmallVector<Attribute> slots;
    for (int64_t row = 0; row < rows; ++row)
      for (int64_t col = 0; col < cols; ++col) {
        SlotAttr slot = memorySlot(op, access, row, col, role,
                                   access.vector() ? VectorLayout::rows
                                                   : VectorLayout::none,
                                   /*transposed=*/false, tileDtype);
        if (!slot)
          return false;
        slots.push_back(number(slot, input));
      }
    setSlots(op, slots);
    return true;
  }

  // Constants that are not splats (the clamp masks) are read from DRAM: the
  // constant op carries one constant slot per tile (row-major grid order), each
  // holding the tile's raw BF16 bits; one tile is a bare slot.
  bool assignConstants() {
    SmallVector<arith::ConstantOp> constants;
    func.walk([&](linalg::GenericOp generic) {
      for (Value input : generic.getDpsInputs()) {
        auto slice = input.getDefiningOp<tensor::ExtractSliceOp>();
        Operation *source =
            slice ? slice.getSource().getDefiningOp() : input.getDefiningOp();
        if (auto constant = dyn_cast_or_null<arith::ConstantOp>(source))
          if (!llvm::is_contained(constants, constant))
            constants.push_back(constant);
      }
    });
    Builder builder(func.getContext());
    for (arith::ConstantOp constant : constants) {
      auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
      if (!dense || dense.isSplat() ||
          !cast<RankedTensorType>(dense.getType()).getElementType().isBF16())
        continue;
      auto type = cast<RankedTensorType>(dense.getType());
      bool grid = type.getRank() == 4 && type.getDimSize(2) == kTile &&
                  type.getDimSize(3) == kTile;
      if (!grid && type.getShape() != ArrayRef<int64_t>{kTile, kTile})
        return reject(constant, "a constant must be one tile or a grid of "
                                "tiles");
      SmallVector<APInt> bits;
      for (APFloat value : llvm::make_range(dense.value_begin<APFloat>(),
                                            dense.value_end<APFloat>()))
        bits.push_back(value.bitcastToAPInt());
      auto patternType =
          RankedTensorType::get({kTile, kTile}, builder.getIntegerType(16));
      int64_t tiles = grid ? type.getDimSize(0) * type.getDimSize(1) : 1;
      SmallVector<Attribute> slots;
      for (int64_t tile = 0; tile < tiles; ++tile) {
        auto pattern = DenseIntElementsAttr::get(
            patternType,
            ArrayRef<APInt>(bits).slice(tile * kTile * kTile, kTile * kTile));
        SmallVector<int64_t> box{0, 0, kTile, kTile};
        auto emitError = [&]() { return constant->emitError(); };
        SlotAttr slot = SlotAttr::getChecked(
            llvm::function_ref<InFlightDiagnostic()>(emitError),
            constant->getContext(), int32_t(0), SlotKind::constant, int32_t(0),
            SlotRole::mask_bf16, int64_t(0), int64_t(0), int64_t(0),
            ArrayRef<int64_t>(box), ArrayRef<int64_t>(box), FloatAttr(),
            pattern, VectorLayout::none, false, Type(builder.getBF16Type()),
            Type(builder.getBF16Type()));
        if (!slot)
          return false;
        slots.push_back(number(slot, /*input=*/true));
      }
      setSlots(constant, slots);
    }
    return true;
  }

  bool run() {
    if (!func.getBody().hasOneBlock())
      return reject(func, "slot assignment requires a single straight-line "
                          "entry block");
    SmallVector<Operation *> memoryOps;
    bool ok = true;
    func.walk([&](Operation *op) {
      if (!isa<LoadOp, StoreOp>(op))
        return;
      memoryOps.push_back(op);
      if (std::optional<Access> access = analyze(op))
        accesses.try_emplace(op, *access);
      else
        ok = false;
    });
    if (!ok || !checkDependences(memoryOps))
      return false;

    // The tiles a pack yields, to tell loaded vectors from computed ones.
    DenseSet<Operation *> loadedTiles;
    SmallVector<linalg::PackOp> packs;
    SmallVector<linalg::UnPackOp> unpacks;
    func.walk([&](Operation *op) {
      if (auto pack = dyn_cast<linalg::PackOp>(op))
        packs.push_back(pack);
      else if (auto unpack = dyn_cast<linalg::UnPackOp>(op))
        unpacks.push_back(unpack);
    });
    for (linalg::PackOp pack : packs)
      for (Operation *user : pack->getUsers())
        loadedTiles.insert(user);
    layouts = computeVectorLayouts(func, [&](Value value) {
      return loadedTiles.contains(value.getDefiningOp());
    });
    // shortcut: a computed vector needed both as rows and as columns has one
    // value per lane, so it would need a transpose of a vector; upgrade when a
    // kernel needs it.
    if (!layouts.conflicts.empty())
      return reject(layouts.conflicts.front() ? layouts.conflicts.front()
                                              : func.getOperation(),
                    "a computed vector needed both as rows and as columns");

    for (linalg::PackOp pack : packs)
      if (!assignInputs(pack))
        return false;
    for (size_t i = 0; i < extraPacks.size(); ++i)
      if (!assignInputs(extraPacks[i]))
        return false;
    for (Operation *op : memoryOps) {
      auto load = dyn_cast<LoadOp>(op);
      bool packed =
          load && llvm::any_of(load->getUsers(), [](Operation *user) {
            return isa<linalg::PackOp, arith::ExtFOp, arith::TruncFOp>(user);
          });
      if (load && !packed && !load->use_empty() && !assignCopy(op, true))
        return false;
    }
    for (linalg::UnPackOp unpack : unpacks)
      if (!assignOutputs(unpack))
        return false;
    for (Operation *op : memoryOps)
      if (auto store = dyn_cast<StoreOp>(op)) {
        Operation *value = sourceBehindCasts(store.getValue());
        if (isa_and_nonnull<linalg::UnPackOp>(value))
          continue;
        if (!isa_and_nonnull<LoadOp>(value) &&
            !std::getenv("TTATLAS_PLUGIN_RUNTIME"))
          return reject(op, "a stored value that is neither a linalg.unpack "
                            "nor a tt.load");
        if (!assignCopy(op, false))
          return false;
      }
    return assignConstants();
  }
};

struct AssignSlots : impl::AssignSlotsBase<AssignSlots> {
  void runOnOperation() override {
    for (FuncOp func : getOperation().getOps<FuncOp>()) {
      Assigner assigner(func);
      if (!assigner.run())
        return signalPassFailure();
      if (std::getenv("TTATLAS_TRACE_BODY"))
        func.walk([&](linalg::GenericOp op) {
          llvm::errs() << "ttatlas AssignSlots generic body="
                       << std::distance(op.getBody()->begin(), op.getBody()->end())
                       << "\n";
        });
    }
  }
};

} // namespace
} // namespace mlir::triton
