// Pass 2, ttatlas-normalize. Contract: triton-atlas design agenda, page
// "Pass 2 · ttatlas-normalize". Placeholder for now: runOnOperation() does
// nothing. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_NORMALIZE
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct Normalize : impl::NormalizeBase<Normalize> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
