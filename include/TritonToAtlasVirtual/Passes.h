// Declarations of the TTIR -> Atlas virtual ops passes, generated from Passes.td.
// Shape follows triton v3.9.0 third_party/amd/include/TritonAMDGPUToLLVM/Passes.h.
#ifndef TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES_H
#define TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES_H

#include "Atlas/AtlasDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton {

// GEN_PASS_DECL declares each pass's options struct and its create function
// (createConvertTritonToAtlasVirtual()); GEN_PASS_REGISTRATION adds registerTritonToAtlasVirtualPasses().
#define GEN_PASS_DECL
#include "TritonToAtlasVirtual/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TritonToAtlasVirtual/Passes.h.inc"

// Adds our passes, in order, to pm (lib/TritonToAtlasVirtual/Pipeline.cpp).
void buildTritonToAtlasVirtualPipeline(OpPassManager &pm);

} // namespace mlir::triton

#endif // TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES_H
