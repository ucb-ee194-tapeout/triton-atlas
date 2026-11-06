// Pass 4, ttatlas-legalize-shapes. Contract: triton-atlas design agenda, page
// "Pass 4 · ttatlas-legalize-shapes". Placeholder for now: runOnOperation()
// does nothing. Shape follows triton v3.9.0
// third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZESHAPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct LegalizeShapes : impl::LegalizeShapesBase<LegalizeShapes> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
