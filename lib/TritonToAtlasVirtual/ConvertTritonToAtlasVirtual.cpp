// TTIR -> Atlas virtual ops. Empty for now: the base class generated from Passes.td does the pass bookkeeping, and
// runOnOperation() is where the rewriting will go. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONTOATLASVIRTUAL
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

struct ConvertTritonToAtlasVirtual : impl::ConvertTritonToAtlasVirtualBase<ConvertTritonToAtlasVirtual> {
  void runOnOperation() override {}
};

} // namespace

} // namespace mlir::triton
