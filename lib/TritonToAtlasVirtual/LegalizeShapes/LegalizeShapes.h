// The pieces of pass 4 (ttatlas-legalize-shapes), one file each. Private to
// this folder; LegalizeShapes.cpp runs them.
#ifndef TRITON_ATLAS_LEGALIZESHAPES_LEGALIZESHAPES_H
#define TRITON_ATLAS_LEGALIZESHAPES_LEGALIZESHAPES_H

#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir::triton {

// What a load or store mask proves: its true lanes are the leading box
// `bounds` (rank-1 or rank-2, like the accessed tensor), and the padded lanes
// hold `neutral`.
struct MaskBox {
  ttatlas::PadNeutral neutral;
  SmallVector<int64_t> bounds;
};
using MaskBoxes = DenseMap<Operation *, MaskBox>;

// Masks.cpp: reads the load and store masks into `boxes`, keyed by the memory
// op, and rejects a mask that is not a compile-time leading box. Only masks
// that cut a tile short get an entry. Nothing is tagged yet: a tag belongs to
// a tile, and the tiles do not exist before tiling.
LogicalResult readMaskedBoundaries(ModuleOp module, MaskBoxes &boxes);

// Padding.cpp: the pad value a constant stands for (a tile-wide 0, 1, -inf or
// +inf), or nothing.
std::optional<ttatlas::PadNeutral> constantNeutral(Value value);

// Padding.cpp: the value that padded lanes of `op`'s operands must hold. It is
// the identity of the combiner of each reduction over a padded loop (found with
// matchReduction, mapped from arith::getNeutralElement), zero for a matmul and
// for an op that reduces no padded loop. `padded[i]` says whether loop i is
// padded (empty: all are). Returns nothing, with the reason in `why`, when
// there is no single value the host can fill. The static path and the masked
// load path both use it, so they agree.
std::optional<ttatlas::PadNeutral> operationNeutral(Operation *op,
                                                    ArrayRef<bool> padded = {},
                                                    std::string *why = nullptr);

// Padding.cpp: pads `op` up to whole Atlas tiles with its neutral element.
LogicalResult padToAtlasTiles(IRRewriter &rewriter, linalg::LinalgOp op,
                              linalg::LinalgOp &paddedOp,
                              SmallVectorImpl<Value> &replacements);

// Padding.cpp: after tiling, tags every boundary tile that holds padded lanes
// with its own logical shape (always [rows, cols]; a row is [1, n]), taking the
// mask boxes and the sub-tile pads of padToAtlasTiles into account. A tile of
// a load is tagged on the op that yields it (the load, the tensor.pad of a
// small load, or the extract_slice of a larger one); a tile of a stored tensor
// on the op that produces it. A full tile has no tag.
LogicalResult tagBoundaryTiles(ModuleOp module, const MaskBoxes &boxes);

// Tiling.cpp: the MXU policy tag on a matmul.
bool hasConflictingMxu(Operation *op);
bool setMxuZero(Operation *op);

// Tiling.cpp: the SCF tile-and-fuse driver's inputs and the passes after it.
SmallVector<OpFoldResult> atlasTileSizes(Builder &builder, linalg::LinalgOp op,
                                         bool parallelOnly);
bool exceedsTile(Operation *op);
std::optional<scf::SCFTileAndFuseOptions::ControlFnResult>
fuseLinalgProducersOnly(tensor::ExtractSliceOp, OpResult producer, bool);
LogicalResult unrollTileLoops(ModuleOp module);
void freshFillDestinations(ModuleOp module, IRRewriter &rewriter);

} // namespace mlir::triton

#endif
