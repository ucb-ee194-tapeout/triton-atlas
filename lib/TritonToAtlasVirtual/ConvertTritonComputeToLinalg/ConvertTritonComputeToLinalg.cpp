// Pass 3, convert-triton-compute-to-linalg. Contract: triton-atlas design
// agenda, page "Pass 3 · convert-triton-compute-to-linalg". Placeholder for
// now: runOnOperation() does nothing. Shape follows triton v3.9.0
// third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONCOMPUTETOLINALG
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct ConvertTritonComputeToLinalg
    : impl::ConvertTritonComputeToLinalgBase<ConvertTritonComputeToLinalg> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
