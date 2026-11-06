// Shared structural interpretation of the M1 boundary contract. No slot
// numbering or pointer proof lives here; those belong to AssignSlots.
#ifndef TRITON_ATLAS_BOUNDARY_TILES_H
#define TRITON_ATLAS_BOUNDARY_TILES_H

#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>

namespace mlir::triton {
struct BoundaryTile {
  Operation *tag;
  Operation *memory;
  Value value;
  SmallVector<int64_t, 2> offsets;
};

inline bool isBoundaryTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getShape() == ArrayRef<int64_t>{32, 32};
}

template <typename Slice> bool isConstantBoundarySlice(Slice slice) {
  return slice.getStaticOffsets().size() == 2 &&
         llvm::all_of(slice.getStaticOffsets(),
                      [](int64_t v) { return v >= 0; }) &&
         slice.getStaticSizes() == ArrayRef<int64_t>{32, 32} &&
         slice.getStaticStrides() == ArrayRef<int64_t>{1, 1};
}

// Ordered by memory operation, then source slice order (never SSA use-list
// order). Output assemblies must cover every element exactly once.
inline LogicalResult
collectBoundaryTiles(triton::FuncOp function,
                     SmallVectorImpl<BoundaryTile> &tiles,
                     llvm::SmallPtrSetImpl<Operation *> &scaffolding) {
  if (function.isExternal() || function.getBody().getBlocks().size() != 1)
    return ttatlas::emitUnsupported(
        function, "boundary tiles require a defined straight-line function");
  for (Operation &op : function.getBody().front()) {
    auto load = dyn_cast<LoadOp>(op);
    auto store = dyn_cast<StoreOp>(op);
    if (!load && !store)
      continue;
    if ((load && load.getMask()) || (store && store.getMask()))
      return ttatlas::emitUnsupported(
          &op,
          "masked accesses are unsupported; a whole unmasked tile is required");
    if (load && load.getIsVolatile())
      return ttatlas::emitUnsupported(
          &op, "volatile loads cannot be represented by an input slot");
    Value value = load ? load.getResult() : store.getValue();
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (type && type.getRank() == 1)
      return ttatlas::emitUnsupported(
          &op, "1-D bias needs the Stage B driver-broadcast contract");
    if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
        type.getDimSize(0) < 32 || type.getDimSize(1) < 32)
      return ttatlas::emitUnsupported(
          &op,
          "expected a static 32x32 tile or larger rank-two boundary tensor");
    if (isBoundaryTile(type)) {
      tiles.push_back({&op, &op, value, {0, 0}});
      continue;
    }
    scaffolding.insert(&op);
    if (load) {
      SmallVector<tensor::ExtractSliceOp> slices;
      for (Operation *user : load->getUsers()) {
        auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
        if (!slice || !isConstantBoundarySlice(slice) ||
            !isBoundaryTile(slice.getType()) ||
            slice->getBlock() != op.getBlock())
          return ttatlas::emitUnsupported(
              &op, "large tt.load must feed only constant 32x32 extract_slice "
                   "operations");
        for (unsigned axis = 0; axis < 2; ++axis)
          if (slice.getStaticOffsets()[axis] > type.getDimSize(axis) - 32)
            return ttatlas::emitUnsupported(
                &op, "input tile slice is out of bounds");
        slices.push_back(slice);
      }
      llvm::sort(slices, [](auto a, auto b) { return a->isBeforeInBlock(b); });
      for (auto slice : slices)
        tiles.push_back({slice, &op, slice.getResult(),
                         SmallVector<int64_t, 2>(slice.getStaticOffsets())});
    } else {
      if (type.getDimSize(0) % 32 || type.getDimSize(1) % 32)
        return ttatlas::emitUnsupported(
            &op, "stored assembly dimensions must be multiples of 32");
      SmallVector<BoundaryTile> reverse;
      std::set<std::pair<int64_t, int64_t>> covered;
      Operation *consumer = &op;
      while (auto slice = value.getDefiningOp<tensor::InsertSliceOp>()) {
        if (!isConstantBoundarySlice(slice) ||
            !isBoundaryTile(slice.getSource().getType()) ||
            !slice->hasOneUse() || *slice->getUsers().begin() != consumer ||
            slice->getBlock() != op.getBlock())
          return ttatlas::emitUnsupported(&op,
                                          "stored tensor requires an unshared "
                                          "constant tile insert_slice chain");
        auto offsets = slice.getStaticOffsets();
        if (offsets[0] % 32 || offsets[1] % 32 ||
            offsets[0] > type.getDimSize(0) - 32 ||
            offsets[1] > type.getDimSize(1) - 32 ||
            !covered.emplace(offsets[0], offsets[1]).second)
          return ttatlas::emitUnsupported(
              &op,
              "output slices must be disjoint, in bounds and tile aligned");
        reverse.push_back(
            {slice, &op, slice.getSource(), SmallVector<int64_t, 2>(offsets)});
        scaffolding.insert(slice);
        consumer = slice;
        value = slice.getDest();
      }
      auto empty = value.getDefiningOp<tensor::EmptyOp>();
      if (!empty || !empty->hasOneUse() ||
          covered.size() != static_cast<uint64_t>(type.getDimSize(0) / 32) *
                                static_cast<uint64_t>(type.getDimSize(1) / 32))
        return ttatlas::emitUnsupported(
            &op,
            "output insert_slice chain must fully cover a fresh tensor.empty");
      scaffolding.insert(empty);
      llvm::append_range(tiles, llvm::reverse(reverse));
    }
  }
  return success();
}
} // namespace mlir::triton
#endif
