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

namespace mlir::triton {

// Masks.cpp: reads the load and store masks and tags the ops whose padded
// lanes need a neutral value. Fills `storePadTags` with the tag of each store,
// to be moved onto the op that produces the stored value once tiling is done.
LogicalResult
tagMaskedBoundaries(ModuleOp module,
                    DenseMap<Operation *, Attribute> &storePadTags);

// Padding.cpp: the neutral element of an operation or constant.
std::optional<ttatlas::PadNeutral> constantNeutral(Value value);
std::optional<ttatlas::PadNeutral> operationNeutral(Operation *op);

// Padding.cpp: pads `op` up to whole Atlas tiles with its neutral element.
LogicalResult padToAtlasTiles(IRRewriter &rewriter, linalg::LinalgOp op,
                              linalg::LinalgOp &paddedOp,
                              SmallVectorImpl<Value> &replacements);

// Padding.cpp: moves each store's pad tag onto the op that produces its value.
LogicalResult
propagateStorePadTags(const DenseMap<Operation *, Attribute> &storePadTags);

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
