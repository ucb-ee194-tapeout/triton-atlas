// Pass 4, ttatlas-legalize-dtypes. Contract: triton-atlas design agenda, page
// "Pass 4 · ttatlas-legalize-dtypes". Placeholder for now: runOnOperation()
// does nothing. Shape follows triton v3.9.0
// third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZEDTYPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct LegalizeDtypes : impl::LegalizeDtypesBase<LegalizeDtypes> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
