// Shared by the TritonToAtlasVirtual passes: evaluate a compile-time integer or
// i1 tensor (an offset or a mask) to one constant per lane, or report that it
// is dynamic. Offsets and masks stay as op trees in the IR; this is the one
// place that decides what "known at compile time" means.
#ifndef TRITON_TO_ATLAS_VIRTUAL_LANE_VALUES_H
#define TRITON_TO_ATLAS_VIRTUAL_LANE_VALUES_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::triton {

// One constant per lane of a tensor (or one lane for a scalar, empty shape),
// flat in row-major order. Integers are kept sign-extended from their bit
// width; i1 is 0 or 1.
struct LaneValues {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> values;
};

// Lanes above this count are treated as dynamic, to keep the evaluation cheap.
inline constexpr int64_t kMaxEvaluatedLanes = 64 * 1024;

// Evaluate `value` (integer, index or i1; scalar or ranked tensor) per lane.
// std::nullopt means "dynamic": it depends on a block argument, a program id, a
// load, or an op this evaluator does not know.
std::optional<LaneValues> evaluateLanes(Value value);

// A pointer tensor (or scalar pointer) whose lanes are `base + offsets[lane]`
// elements of one kernel argument.
struct PointerLanes {
  BlockArgument base;
  LaneValues offsets;
};

// Follows tt.splat of a pointer block argument, tt.addptr, tt.broadcast and
// tt.expand_dims.
std::optional<PointerLanes> evaluatePointer(Value value);

// If the true lanes of `mask` are exactly the leading box [0,b0) x [0,b1) x ...
// of its shape, return the b's (the whole shape when every lane is true; a zero
// when no lane is true). std::nullopt if the mask is dynamic or its true lanes
// have any other shape.
std::optional<SmallVector<int64_t>> maskBounds(Value mask);

} // namespace mlir::triton

#endif
