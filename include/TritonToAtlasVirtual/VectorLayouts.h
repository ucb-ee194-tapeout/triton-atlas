// The layout of every 32-vector in tile form: along rows (lane i is row i,
// the vector repeated across the columns) or along columns (lane i is column
// i, repeated down the rows). One helper decides it for passes 6 and 7, so a
// loaded vector's slot and the broadcast that reads it agree.
//
// A broadcast generic reading a vector through (d0, d1) -> (d0) needs rows,
// through (d1) columns; a row reduction yields rows; an elementwise generic on
// vectors gives all its operands and its result one layout (union-find). A
// vector nothing constrains is rows. A loaded vector (a tile of a pack) is not
// merged across its readers: each use takes the layout its reader needs, and a
// vector needed both ways is loaded twice, once per layout (pass 6 clones the
// pack and gives each its own slot). A computed vector whose class needs both
// layouts is a conflict, reported to the caller (pass 6 rejects it by name).
#ifndef TRITON_ATLAS_VECTORLAYOUTS_H
#define TRITON_ATLAS_VECTORLAYOUTS_H

#include "TTAtlas/TTAtlas.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton {

struct VectorLayouts {
  // Layout of a computed vector (a generic's result or init), by value.
  DenseMap<Value, ttatlas::VectorLayout> computed;
  // Layout each use of a loaded vector asks for, by use.
  DenseMap<OpOperand *, ttatlas::VectorLayout> uses;
  // Generics whose vectors are needed both ways.
  SmallVector<Operation *> conflicts;

  ttatlas::VectorLayout of(Value value) const;
  ttatlas::VectorLayout ofUse(OpOperand &use) const;
};

// Computes the layouts of every 32-vector under `root`. `isLoaded` says which
// vector values come straight from a slot.
VectorLayouts computeVectorLayouts(Operation *root,
                                   function_ref<bool(Value)> isLoaded);

} // namespace mlir::triton

#endif
