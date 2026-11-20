// Pass 7's emission walk (linalg specialization first, then one
// ordered walk with a map from each value to its tiles). Nothing here rewrites:
// the Atlas function is built beside the tt.func, op by op in program order,
// with the state token threaded through each stateful op as it is emitted, so
// program order is a legal schedule (atlas-mlir reorders independent work
// later).
//
// A value's tiles are cells. A pack gives one pending input cell per slot; the
// virtual_input is emitted when a consumer first needs the tile and reused
// after. Slices and reshapes only move cells around. A K chain's matmuls are
// emitted together at its last matmul, so the MXU ops stay contiguous.
#include "Emitter.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "TTAtlas/AtlasTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton {
namespace {

using ttatlas::SlotAttr;
using ttatlas::VectorLayout;

constexpr unsigned kNone = ~0u;
// atlas-hw: the BF16 bits of -200 (the exp clamp's floor), of 1.0 (a divide
// whose numerator needs no multiply), and the identities of a row sum (+0 or
// -0), max (-inf) and min (+inf).
constexpr uint16_t kMinusTwoHundred = 0xC348, kOne = 0x3F80, kZero = 0x0000,
                   kNegZero = 0x8000, kNegInf = 0xFF80, kPosInf = 0x7F80;
// Pass 7 uses MXU0 for now; the unit would be recorded in the metadata when
// MXU1 is used too.
constexpr int32_t kMxuUnit = 0;
// A lower bound of the words a virtual op lowers to, measured on atlas-mlir's
// lowering of small kernels: a BF16 input 16, an FP8 input 11 (half the bytes),
// an output 15, every other op at least one. Checked on four of the lit tests'
// kernels: the sum of these never exceeds the words atlas-mlir emits.
constexpr unsigned kBF16InputWords = 16, kFP8InputWords = 11, kOutputWords = 15;

// One tile (or one 32-vector). Pending: a slot not yet read. Splat: every
// element is `bits`. Ready: an Atlas value. ChainMid: a K chain's matmul that
// is not the last one yet.
struct Cell {
  enum Kind { Pending, Splat, Ready, ChainMid } kind = Ready;
  SlotAttr slot;
  uint16_t bits = 0;
  Value value;
  Operation *matmul = nullptr;
  bool fp8 = false;
  // The layout of a 32-vector (none for a full tile); a splat fits either.
  VectorLayout layout = VectorLayout::none;
  bool anyLayout = false;
  // The row reduction that made a Ready cell (row_sum, ...), for the
  // reduction-init rule.
  StringRef reduction;
};

// The cells of one value, row-major over its grid of tiles (empty grid: one
// tile). tileRank is 2 for 32x32 tiles and 1 for 32-vectors.
struct Tiles {
  SmallVector<int64_t> grid;
  SmallVector<unsigned> cells;
  int64_t tileRank = 2;
};

// The tiles of a grid.
int64_t tileCount(ArrayRef<int64_t> grid) {
  int64_t count = 1;
  for (int64_t extent : grid)
    count *= extent;
  return count;
}

SmallVector<SlotAttr> slotsOf(Operation *op) {
  SmallVector<SlotAttr> slots;
  Attribute attr = op->getAttr(ttatlas::kSlot);
  if (auto slot = dyn_cast_if_present<SlotAttr>(attr))
    slots.push_back(slot);
  else if (auto array = dyn_cast_if_present<ArrayAttr>(attr))
    for (Attribute item : array)
      slots.push_back(cast<SlotAttr>(item));
  return slots;
}

uint16_t bf16Bits(const APFloat &value) {
  APFloat converted = value;
  bool lost;
  converted.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &lost);
  return converted.bitcastToAPInt().getZExtValue();
}

// Address math and masks: integer, index, pointer and boolean values.
bool isAddressValue(Operation *op) {
  return op->getNumResults() > 0 &&
         llvm::all_of(op->getResultTypes(), [](Type type) {
           return isa<IntegerType, IndexType, PointerType>(
               getElementTypeOrSelf(type));
         });
}

bool isMap(AffineMap map, ArrayRef<unsigned> dims) {
  if (map.getNumResults() != dims.size())
    return false;
  for (auto [result, dim] : llvm::zip(map.getResults(), dims)) {
    auto expr = dyn_cast<AffineDimExpr>(result);
    if (!expr || expr.getPosition() != dim)
      return false;
  }
  return true;
}

struct Emitter {
  FuncOp source;
  SlotTables &tables;
  OpBuilder builder;
  MLIRContext *context;
  Value state;
  unsigned words = 0;
  SmallVector<Cell> cells;
  DenseMap<Value, Tiles> tiles;
  // Host tensors: a tt.load's result and its casts; an unpack's result and its
  // casts. The packer converts them.
  DenseSet<Value> hostIn, hostOut;
  // A compute-free kernel's tile loads (slots on the tt.load) and their casts.
  DenseSet<Value> directTile;
  DenseMap<int32_t, Value> inputValues;
  DenseMap<uint16_t, Value> splatValues;
  Type stateType, bf16Type, fp8Type;

  Emitter(FuncOp source, SlotTables &tables)
      : source(source), tables(tables), builder(source),
        context(source.getContext()) {
    stateType = atlas::VirtualStateType::get(context);
    bf16Type = atlas::VirtualBF16Type::get(context);
    fp8Type = atlas::VirtualFP8Type::get(context);
  }

  LogicalResult reject(Operation *op, const Twine &reason) {
    ttatlas::emitUnsupported(op, reason);
    return failure();
  }

  template <typename Op>
  Op make(Location loc, TypeRange results, ValueRange operands,
          ArrayRef<NamedAttribute> attributes = {}) {
    ++words;
    return Op::create(builder, loc, results, operands, attributes);
  }
  NamedAttribute named(StringRef name, Attribute value) {
    return builder.getNamedAttr(name, value);
  }
  NamedAttribute named(StringRef name, int64_t value) {
    return named(name, builder.getI32IntegerAttr(value));
  }

  // -----------------------------------------------------------------------------------------------
  // cells
  unsigned addCell(Cell cell) {
    cells.push_back(cell);
    return cells.size() - 1;
  }
  Cell ready(Value value, bool fp8 = false) {
    Cell cell;
    cell.value = value;
    cell.fp8 = fp8;
    return cell;
  }
  Cell splat(uint16_t bits) {
    Cell cell;
    cell.kind = Cell::Splat;
    cell.bits = bits;
    cell.anyLayout = true;
    return cell;
  }
  Cell pending(SlotAttr slot) {
    Cell cell;
    cell.kind = Cell::Pending;
    cell.slot = slot;
    cell.fp8 = isa<Float8E4M3FNType>(slot.getTileDtype());
    cell.layout = slot.getVectorLayout();
    return cell;
  }
  Tiles single(unsigned cell, int64_t tileRank) {
    return {{}, {cell}, tileRank};
  }
  void setSingle(Value value, Cell cell) {
    int64_t rank = cast<RankedTensorType>(value.getType()).getRank();
    tiles[value] = single(addCell(cell), rank);
  }

  // The one cell of a single-tile value; a scalar constant stands for a splat
  // tile (in a generic's body, a fill).
  FailureOr<unsigned> tile(Operation *user, Value value) {
    auto found = tiles.find(value);
    if (found != tiles.end() && found->second.cells.size() == 1 &&
        found->second.cells.front() != kNone)
      return found->second.cells.front();
    if (auto constant = value.getDefiningOp<arith::ConstantOp>())
      if (auto scalar = dyn_cast<FloatAttr>(constant.getValue()))
        return addCell(splat(bf16Bits(scalar.getValue())));
    return reject(user, "an operand is not one defined tile (pass 4 leaves one "
                        "tile per compute value)");
  }

  // The Atlas value of a cell, emitting its input or splat the first time.
  FailureOr<Value> materialize(Operation *user, unsigned id) {
    Cell cell = cells[id];
    switch (cell.kind) {
    case Cell::Ready:
      return cell.value;
    case Cell::ChainMid:
      return reject(
          user,
          "a K-chain value is used before the chain ends (mid-chain consumer)");
    case Cell::Splat: {
      Value &known = splatValues[cell.bits];
      if (!known)
        known = make<atlas::VirtualVPUSplatOp>(user->getLoc(), {bf16Type}, {},
                                               {named("bits", cell.bits)})
                    .getResult();
      return known;
    }
    case Cell::Pending: {
      int32_t index = cell.slot.getIndex();
      auto [entry, inserted] = tables.inputs.emplace(index, cell.slot);
      if (!inserted && entry->second != cell.slot)
        return reject(user,
                      "two different slots share input index " + Twine(index));
      Value &known = inputValues[index];
      if (!known) {
        Operation *input = cell.fp8 ? make<atlas::VirtualInputFP8Op>(
                                          user->getLoc(), {stateType, fp8Type},
                                          {state}, {named("index", index)})
                                          .getOperation()
                                    : make<atlas::VirtualInputBF16Op>(
                                          user->getLoc(), {stateType, bf16Type},
                                          {state}, {named("index", index)})
                                          .getOperation();
        words += (cell.fp8 ? kFP8InputWords : kBF16InputWords) - 1;
        state = input->getResult(0);
        known = input->getResult(1);
      }
      return known;
    }
    }
    llvm_unreachable("every cell kind is handled");
  }

  // -----------------------------------------------------------------------------------------------
  // VPU The layout of an op over two cells: a vector's own, or none for tiles;
  // a splat takes the other's.
  LogicalResult unify(Operation *op, const Cell &a, const Cell &b, Cell &out) {
    out.anyLayout = a.anyLayout && b.anyLayout;
    if (a.anyLayout)
      out.layout = b.layout;
    else if (b.anyLayout || a.layout == b.layout)
      out.layout = a.layout;
    else
      return reject(op, "vector layouts conflict: one operand has a value per "
                        "row, the other per column");
    return success();
  }

  FailureOr<Value> bf16Value(Operation *op, unsigned id) {
    if (cells[id].fp8)
      return reject(op, "the VPU works on BF16 tiles; an FP8 tile can only "
                        "feed a matmul or a transpose");
    return materialize(op, id);
  }

  FailureOr<unsigned> binary(Operation *op, StringRef kind, unsigned a,
                             unsigned b) {
    FailureOr<Value> lhs = bf16Value(op, a), rhs = bf16Value(op, b);
    if (failed(lhs) || failed(rhs))
      return failure();
    Cell result;
    if (failed(unify(op, cells[a], cells[b], result)))
      return failure();
    result.value = make<atlas::VirtualVPUBinaryOp>(
                       op->getLoc(), {bf16Type}, {*lhs, *rhs},
                       {named("kind", builder.getStringAttr(kind))})
                       .getResult();
    return addCell(result);
  }

  FailureOr<unsigned> unary(Operation *op, StringRef kind, unsigned a) {
    FailureOr<Value> input = bf16Value(op, a);
    if (failed(input))
      return failure();
    Cell result;
    result.layout = cells[a].layout;
    result.anyLayout = cells[a].anyLayout;
    result.value = make<atlas::VirtualVPUUnaryOp>(
                       op->getLoc(), {bf16Type}, {*input},
                       {named("kind", builder.getStringAttr(kind))})
                       .getResult();
    return addCell(result);
  }

  // x / y is x * recip(y); a numerator of 1 needs no multiply.
  FailureOr<unsigned> divide(Operation *op, unsigned a, unsigned b) {
    bool unit = cells[a].kind == Cell::Splat && cells[a].bits == kOne;
    // Read the numerator first, so inputs are read in the order the source
    // names them.
    if (!unit && failed(bf16Value(op, a)))
      return failure();
    FailureOr<unsigned> reciprocal = unary(op, "recip", b);
    if (failed(reciprocal) || unit)
      return reciprocal;
    return binary(op, "mul", a, *reciprocal);
  }

  // exp and exp2 are wrong on the device for inputs far below zero, so they
  // read max(x, -200) (both give 0 there).
  FailureOr<unsigned> exponential(Operation *op, StringRef kind, unsigned a) {
    unsigned floor = addCell(splat(kMinusTwoHundred));
    FailureOr<unsigned> clamped = binary(op, "max", a, floor);
    if (failed(clamped))
      return failure();
    return unary(op, kind, *clamped);
  }

  // -----------------------------------------------------------------------------------------------
  // boundary
  LogicalResult emitOutput(Operation *op, unsigned id, SlotAttr slot) {
    if (id == kNone)
      return reject(op, "a tile of the stored tensor is never computed");
    Cell cell = cells[id];
    if (cell.fp8)
      return reject(op, "an FP8 tile cannot be stored; outputs are BF16");
    if (!cell.anyLayout && cell.layout != slot.getVectorLayout())
      return reject(op, "the stored tile's layout (one value per row, per "
                        "column, or a full tile) disagrees with its "
                        "slot's vector_layout");
    if (slot.getKind() != ttatlas::SlotKind::memory ||
        !slot.getTileDtype().isBF16() || slot.getTransposed())
      return reject(op, "an output slot must be a memory slot with a bf16 tile "
                        "dtype, not transposed (the packer reads the tile as "
                        "stored)");
    if (!tables.outputs.emplace(slot.getIndex(), slot).second)
      return reject(op,
                    "two stores use output index " + Twine(slot.getIndex()));
    FailureOr<Value> value = materialize(op, id);
    if (failed(value))
      return failure();
    // A tile straight from an input goes through a move, which keeps an input's
    // register out of the DMA store.
    if (cell.kind == Cell::Pending)
      value = make<atlas::VirtualVPUUnaryOp>(
                  op->getLoc(), {bf16Type}, {*value},
                  {named("kind", builder.getStringAttr("mov"))})
                  .getResult();
    state = make<atlas::VirtualOutputBF16Op>(op->getLoc(), {stateType},
                                             {state, *value},
                                             {named("index", slot.getIndex())})
                .getResult();
    words += kOutputWords - 1;
    return success();
  }

  LogicalResult load(LoadOp op) {
    SmallVector<SlotAttr> slots = slotsOf(op);
    if (slots.empty()) {
      hostIn.insert(op.getResult());
      return success();
    }
    // A compute-free kernel: the slots sit on the load itself, one per tile.
    auto type = cast<RankedTensorType>(op.getType());
    Tiles result;
    result.tileRank = type.getRank();
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      result.grid.push_back(llvm::divideCeil(type.getDimSize(axis), 32));
    if (static_cast<int64_t>(slots.size()) != tileCount(result.grid))
      return reject(op, "the load carries " + Twine(slots.size()) +
                            " slots for " + Twine(tileCount(result.grid)) +
                            " tiles");
    for (SlotAttr slot : slots)
      result.cells.push_back(addCell(pending(slot)));
    tiles[op.getResult()] = result;
    directTile.insert(op.getResult());
    return success();
  }

  LogicalResult store(StoreOp op) {
    Value value = op.getValue();
    if (hostOut.contains(value))
      return success();
    SmallVector<SlotAttr> slots = slotsOf(op);
    auto found = tiles.find(value);
    if (found == tiles.end() || slots.empty() ||
        slots.size() != found->second.cells.size())
      return reject(op, "a store needs a linalg.unpack with slots, or one slot "
                        "per tile on a direct tile store");
    for (auto [id, slot] : llvm::zip(found->second.cells, slots))
      if (failed(emitOutput(op, id, slot)))
        return failure();
    return success();
  }

  // A cast between host dtypes belongs to the host packer; a cast between tiles
  // is pass 5's to avoid.
  LogicalResult cast(Operation *op, Value input, Value result) {
    if (hostIn.contains(input))
      hostIn.insert(result);
    else if (hostOut.contains(input))
      hostOut.insert(result);
    else if (directTile.contains(input)) {
      // Copy first: inserting the result may grow the map and move `input`'s
      // entry.
      Tiles copy = tiles.find(input)->second;
      tiles[result] = std::move(copy);
      directTile.insert(result);
    } else
      return reject(op, "a cast between compute tiles: pass 5 types every "
                        "compute value, so only the casts at the host "
                        "boundary remain");
    return success();
  }

  LogicalResult pack(linalg::PackOp op) {
    int64_t tileRank = op.getStaticInnerTiles().size();
    ArrayRef<int64_t> shape = op.getDest().getType().getShape();
    SmallVector<int64_t> grid(shape.begin(), shape.end() - tileRank);
    int64_t count = 1;
    for (int64_t extent : grid)
      count *= extent;
    Tiles result{grid, SmallVector<unsigned>(count, kNone), tileRank};
    if (!op.getSource().getDefiningOp<tensor::EmptyOp>()) {
      SmallVector<SlotAttr> slots = slotsOf(op);
      if (!hostIn.contains(op.getSource()))
        return reject(op, "a pack of anything but a load (pass 4 folds packs "
                          "of empty or constant tensors)");
      if (static_cast<int64_t>(slots.size()) != count)
        return reject(op, "the pack carries " + Twine(slots.size()) +
                              " slots for " + Twine(count) + " tiles");
      for (auto [cell, slot] : llvm::zip(result.cells, slots))
        cell = addCell(pending(slot));
    }
    tiles[op.getResult()] = result;
    return success();
  }

  LogicalResult unpack(linalg::UnPackOp op) {
    auto found = tiles.find(op.getSource());
    if (found == tiles.end())
      return reject(op, "an unpack of a value that has no tiles");
    SmallVector<SlotAttr> slots = slotsOf(op);
    if (slots.size() != found->second.cells.size())
      return reject(op, "the unpack carries " + Twine(slots.size()) +
                            " slots for " + Twine(found->second.cells.size()) +
                            " tiles");
    for (auto [id, slot] : llvm::zip(found->second.cells, slots))
      if (failed(emitOutput(op, id, slot)))
        return failure();
    hostOut.insert(op.getResult());
    return success();
  }

  // -----------------------------------------------------------------------------------------------
  // slices, reshapes The cells of a region of a grid, row-major. offsets and
  // sizes cover the grid dims; the tile dims must be whole.
  FailureOr<SmallVector<unsigned>> region(Operation *op, const Tiles &source,
                                          ArrayRef<int64_t> offsets,
                                          ArrayRef<int64_t> sizes,
                                          ArrayRef<int64_t> strides,
                                          ArrayRef<int64_t> tileShape) {
    size_t gridRank = source.grid.size();
    if (offsets.size() != gridRank + source.tileRank ||
        llvm::any_of(offsets, ShapedType::isDynamic) ||
        llvm::any_of(sizes, ShapedType::isDynamic) ||
        !llvm::all_of(strides, [](int64_t s) { return s == 1; }))
      return reject(op, "a slice of tiles must be static with unit strides");
    for (int64_t axis = 0; axis < source.tileRank; ++axis)
      if (offsets[gridRank + axis] != 0 ||
          sizes[gridRank + axis] != tileShape[gridRank + axis])
        return reject(op, "a slice that cuts into a tile");
    SmallVector<unsigned> out;
    SmallVector<int64_t> at(gridRank, 0);
    int64_t count = 1;
    for (size_t axis = 0; axis < gridRank; ++axis)
      count *= sizes[axis];
    for (int64_t n = 0; n < count; ++n) {
      int64_t flat = 0;
      for (size_t axis = 0; axis < gridRank; ++axis) {
        if (offsets[axis] + at[axis] >= source.grid[axis])
          return reject(op, "a slice outside its tile grid");
        flat = flat * source.grid[axis] + offsets[axis] + at[axis];
      }
      out.push_back(flat);
      for (size_t axis = gridRank; axis-- > 0;) {
        if (++at[axis] < sizes[axis])
          break;
        at[axis] = 0;
      }
    }
    return out;
  }

  Tiles regrid(const Tiles &source, SmallVector<unsigned> cellsOut,
               RankedTensorType type) {
    int64_t gridRank = std::max<int64_t>(0, type.getRank() - source.tileRank);
    return {SmallVector<int64_t>(type.getShape().begin(),
                                 type.getShape().begin() + gridRank),
            std::move(cellsOut), source.tileRank};
  }

  LogicalResult extractSlice(tensor::ExtractSliceOp op) {
    auto found = tiles.find(op.getSource());
    if (found == tiles.end())
      return reject(op, "a slice of a value that has no tiles");
    const Tiles &source = found->second;
    if (!op.getSourceType().hasStaticShape() || !op.getType().hasStaticShape())
      return reject(op, "a slice of tiles must be static");
    FailureOr<SmallVector<unsigned>> flat =
        region(op, source, op.getStaticOffsets(), op.getStaticSizes(),
               op.getStaticStrides(), op.getSourceType().getShape());
    if (failed(flat))
      return failure();
    SmallVector<unsigned> picked;
    for (unsigned index : *flat)
      picked.push_back(source.cells[index]);
    Tiles result = regrid(source, std::move(picked), op.getType());
    int64_t count = 1;
    for (int64_t extent : result.grid)
      count *= extent;
    if (count != static_cast<int64_t>(result.cells.size()))
      return reject(op, "a rank-reducing slice that drops a grid dimension of "
                        "more than one tile");
    tiles[op.getResult()] = result;
    return success();
  }

  LogicalResult insertSlice(tensor::InsertSliceOp op) {
    auto src = tiles.find(op.getSource());
    if (src == tiles.end())
      return reject(op, "an insert of a value that has no tiles");
    Tiles result;
    if (auto dest = tiles.find(op.getDest()); dest != tiles.end()) {
      result = dest->second;
    } else if (op.getDest().getDefiningOp<tensor::EmptyOp>() &&
               op.getDestType().hasStaticShape()) {
      ArrayRef<int64_t> shape = op.getDestType().getShape();
      result.tileRank = src->second.tileRank;
      result.grid.assign(shape.begin(), shape.end() - result.tileRank);
      int64_t count = 1;
      for (int64_t extent : result.grid)
        count *= extent;
      result.cells.assign(count, kNone);
    } else {
      return reject(op, "an insert into a value that has no tiles");
    }
    FailureOr<SmallVector<unsigned>> flat =
        region(op, result, op.getStaticOffsets(), op.getStaticSizes(),
               op.getStaticStrides(), op.getDestType().getShape());
    if (failed(flat) || flat->size() != src->second.cells.size())
      return failed(flat) ? failure()
                          : reject(op, "an insert whose region and source hold "
                                       "different numbers of tiles");
    for (auto [index, cell] : llvm::zip(*flat, src->second.cells))
      result.cells[index] = cell;
    tiles[op.getResult()] = result;
    return success();
  }

  // A reshape keeps the cells; it may only add or drop grid dims of extent 1.
  LogicalResult reshape(Operation *op, Value input, Value output) {
    auto found = tiles.find(input);
    if (found == tiles.end())
      return reject(op, "a reshape of a value that has no tiles");
    Tiles result = regrid(found->second, found->second.cells,
                          cast<RankedTensorType>(output.getType()));
    int64_t count = 1;
    for (int64_t extent : result.grid)
      count *= extent;
    if (count != static_cast<int64_t>(result.cells.size()))
      return reject(op, "a reshape that merges tiles; pass 4 only adds or "
                        "drops grid dimensions of extent 1");
    tiles[output] = result;
    return success();
  }

  // -----------------------------------------------------------------------------------------------
  // compute
  LogicalResult fill(linalg::FillOp op) {
    FailureOr<unsigned> value = tile(op, op.getInputs().front());
    if (failed(value) || cells[*value].kind != Cell::Splat)
      return failed(value)
                 ? failure()
                 : reject(op, "a fill with a value that is not a constant");
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    if (type.getRank() > 2)
      return reject(op, "a fill of several tiles");
    tiles[op.getResult(0)] = single(*value, type.getRank());
    return success();
  }

  // A constant tile, a grid of constant tiles (pass 4's clamp masks, shape
  // [rows, cols, 32, 32]) or a 32-vector. A non-splat one is read from DRAM:
  // its slots are one per tile.
  LogicalResult constant(arith::ConstantOp op) {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type || !isa<FloatType>(type.getElementType()))
      return success(); // scalars and address constants
    // A load's `other` is already in its slot; pass 4 leaves the constant
    // behind as that operand.
    if (llvm::all_of(op->getUses(), [](OpOperand &use) {
          auto load = dyn_cast<LoadOp>(use.getOwner());
          return load && load.getOther() == use.get();
        }))
      return success();
    int64_t tileRank = type.getRank() == 1 ? 1 : 2;
    bool tiled =
        (type.getRank() == 1 || type.getRank() == 2 || type.getRank() == 4) &&
        llvm::all_of(type.getShape().take_back(tileRank),
                     [](int64_t extent) { return extent == 32; }) &&
        (type.getRank() != 2 || type.getDimSize(0) == 32);
    if (!tiled)
      return reject(op, "a constant tensor that is not one 32x32 tile, one "
                        "32-vector or a grid of 32x32 tiles");
    Tiles result;
    result.tileRank = tileRank;
    if (type.getRank() == 4)
      result.grid.assign(type.getShape().begin(), type.getShape().begin() + 2);
    SmallVector<SlotAttr> slots = slotsOf(op);
    if (!slots.empty()) {
      if (slots.front().getKind() != ttatlas::SlotKind::constant ||
          !slots.front().getTileDtype().isBF16())
        return reject(op, "a constant tile's slot must be a constant slot with "
                          "a bf16 tile dtype");
      // Cleanup merges equal constants and keeps one slot, so a merged use
      // would silently read the wrong tile.
      if (type.getRank() != 4 && !op.getResult().hasOneUse())
        return reject(op, "a constant with a slot must have exactly one use; "
                          "run no cleanup after assign-slots");
      if (static_cast<int64_t>(slots.size()) != tileCount(result.grid))
        return reject(op, "the constant carries " + Twine(slots.size()) +
                              " slots for " + Twine(tileCount(result.grid)) +
                              " tiles");
      for (SlotAttr slot : slots)
        result.cells.push_back(addCell(pending(slot)));
      tiles[op.getResult()] = result;
      return success();
    }
    auto dense = dyn_cast<DenseFPElementsAttr>(op.getValue());
    if (!dense || !dense.isSplat())
      return reject(op, "a tile constant that is not a splat needs a slot (the "
                        "host writes it)");
    unsigned cell = addCell(splat(bf16Bits(dense.getSplatValue<APFloat>())));
    result.cells.assign(tileCount(result.grid), cell);
    tiles[op.getResult()] = result;
    return success();
  }

  LogicalResult reduction(Operation *op, StringRef kind, Value input,
                          Value init, Value result) {
    FailureOr<unsigned> in = tile(op, input), start = tile(op, init);
    if (failed(in) || failed(start))
      return failure();
    FailureOr<Value> row = bf16Value(op, *in);
    if (failed(row))
      return failure();
    Cell out;
    out.layout = VectorLayout::rows;
    out.reduction = kind;
    out.value = make<atlas::VirtualVPUReduceOp>(
                    op->getLoc(), {bf16Type}, {*row},
                    {named("kind", builder.getStringAttr(kind))})
                    .getResult();
    unsigned reduced = addCell(out);
    const Cell &seed = cells[*start];
    uint16_t identity = kind == "row_max"   ? kNegInf
                        : kind == "row_min" ? kPosInf
                                            : kZero;
    if (seed.kind == Cell::Splat &&
        (seed.bits == identity ||
         (kind == "row_sum" && seed.bits == kNegZero))) {
      // An identity start is dropped: the virtual reduction takes none.
    } else if (seed.kind == Cell::Ready && seed.reduction == kind) {
      // A row spanning several tiles: the partial result passes through the
      // init.
      StringRef combine = kind == "row_max"   ? "max"
                          : kind == "row_min" ? "min"
                                              : "add";
      FailureOr<unsigned> merged = binary(op, combine, reduced, *start);
      if (failed(merged))
        return failure();
      cells[*merged].reduction = kind;
      reduced = *merged;
    } else {
      return reject(
          op, "a reduction must start from its identity (0 for a sum, -inf for "
              "a max, +inf for a min) or from a "
              "previous reduction of the same kind (a row over several tiles)");
    }
    tiles[result] = single(reduced, 1);
    return success();
  }

  enum class Fn {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Recip,
    Exp,
    Exp2,
    Cast,
    Unknown
  };
  static Fn functionOf(Operation *op) {
    if (isa<arith::AddFOp>(op))
      return Fn::Add;
    if (isa<arith::SubFOp>(op))
      return Fn::Sub;
    if (isa<arith::MulFOp>(op))
      return Fn::Mul;
    if (isa<arith::DivFOp>(op))
      return Fn::Div;
    if (isa<arith::MinNumFOp, arith::MinimumFOp>(op))
      return Fn::Min;
    if (isa<arith::MaxNumFOp, arith::MaximumFOp>(op))
      return Fn::Max;
    if (isa<math::ExpOp>(op))
      return Fn::Exp;
    if (isa<math::Exp2Op>(op))
      return Fn::Exp2;
    if (isa<arith::ExtFOp, arith::TruncFOp>(op))
      return Fn::Cast;
    return Fn::Unknown;
  }

  LogicalResult elementwise(Operation *op, Fn fn, ValueRange operands,
                            Value result) {
    SmallVector<unsigned> in;
    for (Value operand : operands) {
      FailureOr<unsigned> id = tile(op, operand);
      if (failed(id))
        return failure();
      in.push_back(*id);
    }
    FailureOr<unsigned> out = failure();
    switch (fn) {
    case Fn::Add:
    case Fn::Sub:
    case Fn::Mul:
    case Fn::Min:
    case Fn::Max:
      out = binary(op,
                   fn == Fn::Add   ? "add"
                   : fn == Fn::Sub ? "sub"
                   : fn == Fn::Mul ? "mul"
                   : fn == Fn::Min ? "min"
                                   : "max",
                   in[0], in[1]);
      break;
    case Fn::Div:
      out = divide(op, in[0], in[1]);
      break;
    case Fn::Recip:
      out = unary(op, "recip", in[0]);
      break;
    case Fn::Exp:
    case Fn::Exp2:
      out = exponential(op, fn == Fn::Exp ? "exp" : "exp2", in[0]);
      break;
    default:
      return reject(op, "no Atlas VPU operation for this");
    }
    if (failed(out))
      return failure();
    tiles[result] =
        single(*out, cast<RankedTensorType>(result.getType()).getRank());
    return success();
  }

  // Pass 4's clamp mask: x' = max(min(x, hi), lo), a pointwise generic whose
  // body is one minnumf then one maxnumf over its three inputs.
  static bool isClamp(linalg::GenericOp op) {
    Block &body = *op.getBody();
    if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1 ||
        !llvm::all_of(op.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return type == utils::IteratorType::parallel;
                      }) ||
        !llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }) ||
        llvm::range_size(body.without_terminator()) != 2)
      return false;
    auto min = dyn_cast<arith::MinNumFOp>(&body.front());
    auto max = dyn_cast<arith::MaxNumFOp>(body.front().getNextNode());
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    return min && max && max.getLhs() == min.getResult() &&
           yield.getNumOperands() == 1 &&
           yield.getOperand(0) == max.getResult() &&
           min.getLhs() == body.getArgument(0) &&
           min.getRhs() == body.getArgument(1) &&
           max.getRhs() == body.getArgument(2);
  }

  LogicalResult clamp(linalg::GenericOp op) {
    SmallVector<unsigned> in;
    for (Value operand : op.getDpsInputs()) {
      FailureOr<unsigned> id = tile(op, operand);
      if (failed(id))
        return failure();
      in.push_back(*id);
    }
    FailureOr<unsigned> low = binary(op, "min", in[0], in[1]);
    if (failed(low))
      return failure();
    FailureOr<unsigned> out = binary(op, "max", *low, in[2]);
    if (failed(out))
      return failure();
    tiles[op.getResult(0)] = single(
        *out, cast<RankedTensorType>(op.getResult(0).getType()).getRank());
    return success();
  }

  LogicalResult generic(linalg::GenericOp op) {
    Block &body = *op.getBody();
    if (isClamp(op))
      return clamp(op);
    Operation *compute = nullptr;
    for (Operation &inner : body.without_terminator()) {
      if (isa<arith::ConstantOp>(inner))
        continue;
      if (compute)
        return reject(op, "a linalg.generic with several operations in its "
                          "body has no Atlas lowering");
      compute = &inner;
    }
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    if (!compute || compute->getNumResults() != 1 ||
        yield.getNumOperands() != 1 ||
        yield.getOperand(0) != compute->getResult(0))
      return reject(op,
                    "a linalg.generic must yield the result of one operation");
    unsigned numInputs = op.getNumDpsInputs();
    SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    Fn fn = functionOf(compute);

    // The body's operands as tensors: block arguments are the generic's inputs,
    // constants are splats.
    SmallVector<Value> operands;
    BlockArgument outArgument =
        op.getNumDpsInits() ? body.getArgument(numInputs) : BlockArgument();
    for (Value operand : compute->getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand);
          argument && argument.getOwner() == &body) {
        if (argument.getArgNumber() >= numInputs)
          operands.push_back(
              op.getDpsInits()[0]); // only a reduction reads its own output
        else
          operands.push_back(op.getDpsInputs()[argument.getArgNumber()]);
      } else {
        operands.push_back(operand);
      }
    }
    if (fn == Fn::Cast) {
      if (numInputs != 1)
        return reject(op, "a cast generic with several inputs");
      return cast(op, op.getDpsInputs()[0], op.getResult(0));
    }
    if (llvm::all_of(iterators, [](utils::IteratorType type) {
          return type == utils::IteratorType::parallel;
        })) {
      for (AffineMap map : maps)
        if (!map.isIdentity())
          return reject(op, "pass 7 lowers only pointwise generics: every "
                            "indexing map must be the identity (a "
                            "broadcast or transposed operand has no lowering)");
      if (llvm::is_contained(compute->getOperands(), outArgument) &&
          outArgument)
        return reject(op, "a pointwise generic that reads its own output");
      return elementwise(op, fn, operands, op.getResult(0));
    }
    // A row reduction: input (d0, d1) -> output (d0), body combines the
    // accumulator with the element.
    if (iterators.size() != 2 ||
        iterators[0] != utils::IteratorType::parallel ||
        iterators[1] != utils::IteratorType::reduction || numInputs != 1 ||
        !isMap(maps[0], {0, 1}) || !isMap(maps[1], {0}))
      return reject(op, "only a row reduction (reduce axis 1 of a 32x32 tile) "
                        "has an Atlas lowering: the VPU reduces "
                        "rows only");
    StringRef kind;
    if (isa<arith::AddFOp>(compute))
      kind = "row_sum";
    else if (isa<arith::MaxNumFOp, arith::MaximumFOp>(compute))
      kind = "row_max";
    else if (isa<arith::MinNumFOp, arith::MinimumFOp>(compute))
      kind = "row_min";
    else
      return reject(
          op, "the reduction's combiner is not an add, a maximum or a minimum");
    BlockArgument inArgument = body.getArgument(0);
    if (!((compute->getOperand(0) == outArgument &&
           compute->getOperand(1) == inArgument) ||
          (compute->getOperand(1) == outArgument &&
           compute->getOperand(0) == inArgument)))
      return reject(op, "the reduction must combine its accumulator with the "
                        "input element, nothing else");
    return reduction(op, kind, op.getDpsInputs()[0], op.getDpsInits()[0],
                     op.getResult(0));
  }

  LogicalResult broadcast(linalg::BroadcastOp op) {
    FailureOr<unsigned> input = tile(op, op.getInput());
    if (failed(input))
      return failure();
    ArrayRef<int64_t> dimensions = op.getDimensions();
    // Dimension 1 added: one value per row. Dimension 0 added: one value per
    // column.
    if (dimensions.size() != 1 ||
        tiles.find(op.getInput())->second.tileRank != 1)
      return reject(op, "only a 32-vector broadcast along one dimension has an "
                        "Atlas lowering");
    VectorLayout wanted =
        dimensions[0] == 1 ? VectorLayout::rows : VectorLayout::cols;
    Cell cell = cells[*input];
    if (!cell.anyLayout && cell.layout != wanted)
      return reject(
          op,
          "the broadcast needs the vector in the other layout (one value per " +
              Twine(wanted == VectorLayout::rows ? "row" : "column") +
              ") than the one it has; pass 6 records the layout of loaded "
              "vectors and a row reduction gives "
              "one value per row");
    cell.layout = VectorLayout::none;
    cell.anyLayout = false;
    cell.reduction = {};
    tiles[op->getResult(0)] = single(addCell(cell), 2);
    return success();
  }

  LogicalResult transpose(linalg::TransposeOp op) {
    FailureOr<unsigned> input = tile(op, op.getInput());
    if (failed(input))
      return failure();
    if (op.getPermutation() != ArrayRef<int64_t>{1, 0})
      return reject(op, "a transpose that is not a 2-D swap");
    if (!cells[*input].fp8)
      return reject(op, "a transpose of a BF16 tile has no Atlas lowering "
                        "(only FP8 tiles go through the XLU)");
    FailureOr<Value> value = materialize(op, *input);
    if (failed(value))
      return failure();
    tiles[op->getResult(0)] =
        single(addCell(ready(make<atlas::VirtualXLUTransposeOp>(
                                 op->getLoc(), {fp8Type}, {*value})
                                 .getResult(),
                             true)),
               2);
    return success();
  }

  // A K chain is its matmuls linked through their accumulator. Its matmuls are
  // emitted together at the last one.
  LogicalResult matmul(linalg::MatmulOp op) {
    Value result = op.getResult(0);
    SmallVector<Operation *> successors;
    bool otherUse = false;
    for (OpOperand &use : result.getUses()) {
      if (isa<linalg::MatmulOp>(use.getOwner()) && use.getOperandNumber() == 2)
        successors.push_back(use.getOwner());
      else
        otherUse = true;
    }
    if (successors.size() > 1)
      return reject(op, "a matmul result seeds two matmuls (a forked K chain)");
    if (successors.size() == 1) {
      if (otherUse)
        return reject(op, "a K-chain value is used before the chain ends "
                          "(mid-chain consumer)");
      Cell mid;
      mid.kind = Cell::ChainMid;
      mid.matmul = op;
      setSingle(result, mid);
      return success();
    }

    SmallVector<linalg::MatmulOp> chain{op};
    Value accumulator = op.getOutputs()[0];
    while (true) {
      FailureOr<unsigned> id = tile(op, accumulator);
      if (failed(id))
        return failure();
      if (cells[*id].kind != Cell::ChainMid) {
        if (cells[*id].kind != Cell::Splat ||
            (cells[*id].bits != kZero && cells[*id].bits != kNegZero))
          return reject(op, "a matmul's accumulator must be a zero fill or the "
                            "previous matmul of the same K chain");
        break;
      }
      chain.push_back(cast<linalg::MatmulOp>(cells[*id].matmul));
      accumulator = chain.back().getOutputs()[0];
    }
    std::reverse(chain.begin(), chain.end());

    // Materialize every operand first: an input is stateful, and the chain must
    // stay contiguous.
    SmallVector<Value> activations, weights;
    for (linalg::MatmulOp step : chain) {
      SmallVector<AffineMap> maps = step.getIndexingMapsArray();
      bool weightNK = isMap(maps[1], {1, 2});
      if (!isMap(maps[0], {0, 2}) || !(weightNK || isMap(maps[1], {2, 1})) ||
          !isMap(maps[2], {0, 1}))
        return reject(
            step,
            "a matmul whose indexing maps are not A[m,k] x B[k,n] or B[n,k]");
      FailureOr<unsigned> a = tile(step, step.getInputs()[0]),
                          b = tile(step, step.getInputs()[1]);
      if (failed(a) || failed(b))
        return failure();
      if (!cells[*a].fp8 || !cells[*b].fp8)
        return reject(step,
                      "matmul operands must be FP8 tiles (legalize-dtypes "
                      "casts them; a BF16 value cannot feed the MXU)");
      FailureOr<Value> activation = materialize(step, *a),
                       weight = materialize(step, *b);
      if (failed(activation) || failed(weight))
        return failure();
      // The MXU takes B as [n,k]; a [k,n] tile goes through the XLU first.
      Value tileValue = *weight;
      if (!weightNK)
        tileValue = make<atlas::VirtualXLUTransposeOp>(step->getLoc(),
                                                       {fp8Type}, {tileValue})
                        .getResult();
      activations.push_back(*activation);
      weights.push_back(tileValue);
    }

    Type weightType = atlas::VirtualMXUWeightType::get(context, kMxuUnit),
         accType = atlas::VirtualMXUAccType::get(context, kMxuUnit);
    Value acc;
    for (auto [index, step] : llvm::enumerate(chain)) {
      Location loc = step->getLoc();
      Operation *load = make<atlas::VirtualMXULoadWeightOp>(
                            loc, {stateType, weightType},
                            {state, weights[index]}, {named("unit", kMxuUnit)})
                            .getOperation();
      state = load->getResult(0);
      Operation *mxu =
          index == 0 ? make<atlas::VirtualMXUResetOp>(
                           loc, {stateType, accType},
                           {state, activations[index], load->getResult(1)})
                           .getOperation()
                     : make<atlas::VirtualMXUAccumulateOp>(
                           loc, {stateType, accType},
                           {state, activations[index], load->getResult(1), acc})
                           .getOperation();
      state = mxu->getResult(0);
      acc = mxu->getResult(1);
    }
    Operation *readout = make<atlas::VirtualMXUReadoutBF16Op>(
                             op->getLoc(), {stateType, bf16Type}, {state, acc})
                             .getOperation();
    state = readout->getResult(0);
    setSingle(result, ready(readout->getResult(1)));
    return success();
  }

  // -----------------------------------------------------------------------------------------------
  // the walk
  LogicalResult visit(Operation *op) {
    if (isa<ReturnOp, tensor::EmptyOp>(op))
      return success();
    // Every compute tile is BF16 (or an FP8 matmul operand); only the host
    // boundary keeps other float types.
    if (!isa<LoadOp, StoreOp, arith::ExtFOp, arith::TruncFOp, FpToFpOp>(op))
      for (Type type : op->getResultTypes())
        if (auto tensor = dyn_cast<RankedTensorType>(type))
          if (isa<FloatType>(tensor.getElementType()) &&
              !tensor.getElementType().isBF16() &&
              !isa<Float8E4M3FNType>(tensor.getElementType()))
            return reject(op, "compute must be BF16 or FP8 (legalize-dtypes "
                              "types every compute value)");
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op))
      return constant(constantOp);
    if (auto loadOp = dyn_cast<LoadOp>(op))
      return load(loadOp);
    if (auto storeOp = dyn_cast<StoreOp>(op))
      return store(storeOp);
    if (isa<arith::ExtFOp, arith::TruncFOp, FpToFpOp>(op))
      return cast(op, op->getOperand(0), op->getResult(0));
    if (isAddressValue(op))
      return success();
    if (auto packOp = dyn_cast<linalg::PackOp>(op))
      return pack(packOp);
    if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op))
      return unpack(unpackOp);
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op))
      return extractSlice(slice);
    if (auto slice = dyn_cast<tensor::InsertSliceOp>(op))
      return insertSlice(slice);
    if (auto reshapeOp = dyn_cast<tensor::CollapseShapeOp>(op))
      return reshape(op, reshapeOp.getSrc(), reshapeOp.getResult());
    if (auto reshapeOp = dyn_cast<tensor::ExpandShapeOp>(op))
      return reshape(op, reshapeOp.getSrc(), reshapeOp.getResult());
    if (auto fillOp = dyn_cast<linalg::FillOp>(op))
      return fill(fillOp);
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      return generic(genericOp);
    if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op))
      return matmul(matmulOp);
    if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op))
      return broadcast(broadcastOp);
    if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op))
      return transpose(transposeOp);
    if (isa<linalg::AddOp>(op))
      return elementwise(op, Fn::Add, cast<linalg::AddOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::SubOp>(op))
      return elementwise(op, Fn::Sub, cast<linalg::SubOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::MulOp>(op))
      return elementwise(op, Fn::Mul, cast<linalg::MulOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::DivOp>(op))
      return elementwise(op, Fn::Div, cast<linalg::DivOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::MaxOp>(op))
      return elementwise(op, Fn::Max, cast<linalg::MaxOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::MinOp>(op))
      return elementwise(op, Fn::Min, cast<linalg::MinOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::ReciprocalOp>(op))
      return elementwise(op, Fn::Recip,
                         cast<linalg::ReciprocalOp>(op).getInputs(),
                         op->getResult(0));
    if (isa<linalg::ExpOp>(op))
      return elementwise(op, Fn::Exp, cast<linalg::ExpOp>(op).getInputs(),
                         op->getResult(0));
    return reject(op, "pass 7 has no Atlas lowering for this operation");
  }

  FailureOr<func::FuncOp> run() {
    if (source.isExternal() || !source.getBody().hasOneBlock() ||
        source.getFunctionType().getNumResults() != 0) {
      (void)reject(source,
                   "pass 7 needs one defined, single-block, void tt.func");
      return failure();
    }
    auto function =
        func::FuncOp::create(builder, source.getLoc(), source.getName(),
                             FunctionType::get(context, {}, {stateType}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    state = make<atlas::VirtualStartOp>(source.getLoc(), {stateType}, {})
                .getResult();
    for (Operation &op : source.getBody().front()) {
      if (failed(visit(&op))) {
        function.erase();
        return failure();
      }
    }
    func::ReturnOp::create(builder, source.getLoc(), state);
    if (words > static_cast<unsigned>(ttatlas::target::kProgramWords)) {
      (void)reject(
          source,
          "kernel too large for the pinned instruction memory: at least " +
              Twine(words) + " words of " +
              Twine(ttatlas::target::kProgramWords) +
              "; fewer or smaller tiles, or a newer atlas-mlir");
      function.erase();
      return failure();
    }
    return function;
  }
};

} // namespace

FailureOr<func::FuncOp> emitAtlasFunction(FuncOp source, SlotTables &tables) {
  return Emitter(source, tables).run();
}

} // namespace mlir::triton
