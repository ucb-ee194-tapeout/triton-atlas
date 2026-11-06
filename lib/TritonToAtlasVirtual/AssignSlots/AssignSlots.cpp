// Pass 6, ttatlas-assign-slots. Contract: triton-atlas design agenda, page
// "Pass 6 · ttatlas-assign-slots". Placeholder for now: runOnOperation() does
// nothing. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_ASSIGNSLOTS
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct AssignSlots : impl::AssignSlotsBase<AssignSlots> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
