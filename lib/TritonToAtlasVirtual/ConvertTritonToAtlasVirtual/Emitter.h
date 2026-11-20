// Pass 7's emission walk: one ordered pass over a stage-6 tt.func that builds
// the argument-free Atlas function (Emitter.cpp), and what it learns about the
// boundary for the kernel metadata (KernelMetadata.cpp).
#ifndef TRITON_ATLAS_CONVERTTRITONTOATLASVIRTUAL_EMITTER_H
#define TRITON_ATLAS_CONVERTTRITONTOATLASVIRTUAL_EMITTER_H

#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <map>

namespace mlir::triton {

// The slot of every input and output the function reads or writes, by slot
// index.
struct SlotTables {
  std::map<int32_t, ttatlas::SlotAttr> inputs, outputs;
};

// Builds the Atlas function for `source` before `source` in its module, and
// fills `tables`. On a rejection it has emitted the named error and returns
// failure; `source` is specialized first, so it may change either way.
FailureOr<func::FuncOp> emitAtlasFunction(FuncOp source, SlotTables &tables);

// Adds the DRAM bases and ttatlas.kernel (JSON, schema v2) to `function`.
LogicalResult attachKernelMetadata(func::FuncOp function,
                                   const SlotTables &tables);

} // namespace mlir::triton

#endif // TRITON_ATLAS_CONVERTTRITONTOATLASVIRTUAL_EMITTER_H
