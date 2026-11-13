// Pass 1, ttatlas-validate-ttir. Contract: triton-atlas design agenda, page
// "Pass 1 · ttatlas-validate-ttir". For now it only records each load's and
// store's host dtype; it rejects nothing yet. Shape follows triton v3.9.0
// third_party/amd/lib/TritonAMDGPUToLLVM.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_VALIDATETTIR
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;

// The element type a load or store moves in host memory: the pointee of its
// pointer operand (one pointer or a tensor of them). Triton's type verifier
// guarantees an integer or float pointee.
Type pointee(Value ptr) {
  return cast<PointerType>(getElementTypeOrSelf(ptr.getType()))
      .getPointeeType();
}

// Recorded now because pass 5 retypes values; the driver reads it to convert
// at the boundary (e.g. an f32 output Atlas writes as BF16).
void recordHostDtypes(ModuleOp module) {
  module.walk([](LoadOp load) {
    load->setAttr(tta::kHostDtype, TypeAttr::get(pointee(load.getPtr())));
  });
  module.walk([](StoreOp store) {
    store->setAttr(tta::kHostDtype, TypeAttr::get(pointee(store.getPtr())));
  });
}

struct ValidateTTIR : impl::ValidateTTIRBase<ValidateTTIR> {
  void runOnOperation() override { recordHostDtypes(getOperation()); }
};

} // namespace

} // namespace mlir::triton
