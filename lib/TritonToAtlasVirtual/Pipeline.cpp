// The order of our passes lives here, not in Python, so it can run without
// Triton's stages. Same idea as atlas-mlir's lib/AtlasPipeline.cpp. Order and
// reasons: triton-atlas design agenda, "Why these passes, in this order".
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::triton {

// canonicalize + CSE after passes 2, 4 and 5, so dead address math and the
// leftovers of tiling never reach ttatlas-assign-slots. Our tags survive CSE
// (it compares attributes); none runs after pass 7, whose matmul and pack ops
// must keep their order.
static void addCleanup(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void buildTritonToAtlasVirtualPipeline(OpPassManager &pm) {
  pm.addPass(createValidateTTIR()); // 1
  pm.addPass(createNormalize());    // 2
  addCleanup(pm);
  pm.addPass(createConvertTritonComputeToLinalg()); // 3
  pm.addPass(createLegalizeShapes());               // 4
  addCleanup(pm);
  pm.addPass(createLegalizeDtypes()); // 5
  addCleanup(pm);
  pm.addPass(createAssignSlots());                 // 6
  pm.addPass(createConvertTritonToAtlasVirtual()); // 7
  // Pass 8 (atlas-mlir's verify-atlas-virtual-stream) joins once stage A
  // produces virtual IR; while the passes are empty it would reject every
  // kernel.
}

} // namespace mlir::triton
