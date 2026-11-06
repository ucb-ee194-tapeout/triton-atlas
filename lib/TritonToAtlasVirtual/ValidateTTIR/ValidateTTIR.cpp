// Pass 1, ttatlas-validate-ttir. Contract: triton-atlas design agenda, page
// "Pass 1 · ttatlas-validate-ttir". Placeholder for now: runOnOperation() does
// nothing. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_VALIDATETTIR
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct ValidateTTIR : impl::ValidateTTIRBase<ValidateTTIR> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
