// The tile ops of a packed memory boundary (see PackTiles.h).
#include "TritonToAtlasVirtual/PackTiles.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::triton {
namespace {

constexpr int64_t kTile = ttatlas::target::kTileSize;

// The grid coordinate of a slice of a packed tensor that takes one whole tile
// (unit grid sizes, a full tile, unit strides, constant offsets, tile offsets
// zero), or nothing.
template <typename Slice>
std::optional<PackTile> tileSlice(Slice slice, int64_t gridRank) {
  ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  ArrayRef<int64_t> sizes = slice.getStaticSizes();
  if (offsets.size() != sizes.size() ||
      llvm::any_of(slice.getStaticStrides(),
                   [](int64_t stride) { return stride != 1; }) ||
      llvm::any_of(offsets, ShapedType::isDynamic))
    return std::nullopt;
  for (auto [dim, size] : llvm::enumerate(sizes))
    if (size != (int64_t(dim) < gridRank ? 1 : kTile) ||
        (int64_t(dim) >= gridRank && offsets[dim] != 0))
      return std::nullopt;
  return PackTile{slice, offsets[0], gridRank > 1 ? offsets[1] : 0};
}

// A collapse_shape / expand_shape that joins a one-tile grid with its tile.
bool isOneTileReshape(Value packed) {
  auto type = cast<RankedTensorType>(packed.getType());
  int64_t gridRank = type.getRank() / 2;
  return llvm::all_of(type.getShape().take_front(gridRank),
                      [](int64_t size) { return size == 1; });
}

} // namespace

std::optional<SmallVector<PackTile>> inputTiles(linalg::PackOp pack) {
  int64_t gridRank = pack.getDestType().getRank() / 2;
  SmallVector<PackTile> tiles;
  for (Operation *user : pack->getUsers()) {
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(user)) {
      if (auto tile = tileSlice(slice, gridRank)) {
        tiles.push_back(*tile);
        continue;
      }
    } else if (isa<tensor::CollapseShapeOp>(user) &&
               isOneTileReshape(pack.getResult())) {
      tiles.push_back(PackTile{user, 0, 0});
      continue;
    }
    ttatlas::emitUnsupported(user, "a packed load must be read as constant "
                                   "32x32 tiles (or one collapsed tile)");
    return std::nullopt;
  }
  llvm::sort(tiles, [](const PackTile &a, const PackTile &b) {
    return a.op->isBeforeInBlock(b.op);
  });
  return tiles;
}

std::optional<SmallVector<PackTile>> outputTiles(linalg::UnPackOp unpack) {
  auto type = unpack.getSourceType();
  int64_t gridRank = type.getRank() / 2;
  int64_t rows = type.getDimSize(0),
          cols = gridRank > 1 ? type.getDimSize(1) : 1;
  SmallVector<std::optional<PackTile>> grid(rows * cols);
  Value source = unpack.getSource();
  if (auto expand = source.getDefiningOp<tensor::ExpandShapeOp>();
      expand && isOneTileReshape(source)) {
    grid[0] = PackTile{expand, 0, 0};
  } else {
    while (auto insert = source.getDefiningOp<tensor::InsertSliceOp>()) {
      auto tile = tileSlice(insert, gridRank);
      if (!tile || tile->row >= rows || tile->col >= cols) {
        ttatlas::emitUnsupported(unpack, "a stored tensor must be built from "
                                         "constant 32x32 tile inserts");
        return std::nullopt;
      }
      auto &entry = grid[tile->row * cols + tile->col];
      if (!entry)
        entry = tile;
      source = insert.getDest();
    }
  }
  SmallVector<PackTile> tiles;
  for (auto &entry : grid) {
    if (!entry) {
      ttatlas::emitUnsupported(unpack, "a stored tile is never written");
      return std::nullopt;
    }
    tiles.push_back(*entry);
  }
  return tiles;
}

} // namespace mlir::triton
