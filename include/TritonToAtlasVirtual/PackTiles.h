// The tiles of a packed memory boundary after pass 4: where a linalg.pack
// result is read as 32x32 tiles (or 32-vectors) and where a linalg.unpack
// source is written tile by tile. Pass 6 puts one slot on each tile op, and
// the stage-6 contract checks it. Layout of the forms pass 4 leaves:
//   - pack of a tt.load: each tile is a tensor.extract_slice of the pack at
//     constant grid offsets [i, j, 0, 0] (rank-reduced to the tile), or, when
//     the grid is one tile, a tensor.collapse_shape of the whole pack;
//   - unpack for a tt.store: each tile enters the packed tensor through a
//     tensor.insert_slice at [i, j, 0, 0] (the insert nearest the unpack wins
//     where an earlier one wrote an accumulator), or, for a one-tile grid,
//     through a tensor.expand_shape.
// A rank-1 value packs to [G, 32], with offsets [i, 0] and the same shapes.
#ifndef TRITON_ATLAS_PACKTILES_H
#define TRITON_ATLAS_PACKTILES_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton {

struct PackTile {
  Operation *op;            // yields the tile (input) or consumes it (output)
  int64_t row = 0, col = 0; // grid coordinate
};

// Every tile op that reads the pack, in program order. A user of the pack that
// is not a constant tile read, or a tile read twice at different forms, is an
// error reported at that user (nullopt).
std::optional<SmallVector<PackTile>> inputTiles(linalg::PackOp pack);

// The tile op that writes each grid coordinate of the unpack's source, in
// row-major grid order. Reports at the unpack and returns nullopt when a
// coordinate is not written by a constant tile insert (or the source is not
// built from tiles).
std::optional<SmallVector<PackTile>> outputTiles(linalg::UnPackOp unpack);

} // namespace mlir::triton

#endif
