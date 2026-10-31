// The order of our passes lives here, not in Python, so it can run without
// Triton's stages. Same idea as atlas-mlir's lib/AtlasPipeline.cpp.
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::triton {

void buildTritonToAtlasVirtualPipeline(OpPassManager &pm) {
  pm.addPass(createConvertTritonToAtlasVirtual());
}

} // namespace mlir::triton
