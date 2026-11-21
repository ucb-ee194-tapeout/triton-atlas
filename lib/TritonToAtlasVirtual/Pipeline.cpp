// The order of our passes lives here, not in Python, so it can run without
// Triton's stages. Same idea as atlas-mlir's lib/AtlasPipeline.cpp. Order: the
// kernel is validated and normalized, lowered to linalg, retyped, tiled, given
// its DRAM slots, then converted to Atlas virtual IR.
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::triton {

// canonicalize + CSE after passes 2, 5 and 4, so dead address math and the
// leftovers of tiling never reach ttatlas-assign-slots. None runs after pass 6:
// cleanup merges equal constants and keeps one slot for them (pass 7 rejects
// that), and pass 7 needs its matmul and pack ops in order.
static void addCleanup(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

// Pass 4: pack, fold, tile the block grid, fold unit dimensions, with
// canonicalize and CSE after each step.
void buildLegalizeShapesPipeline(OpPassManager &pm) {
  pm.addPass(createPackBlocks()); // 4a
  addCleanup(pm);
  pm.addPass(createFoldPackPairs()); // 4b
  addCleanup(pm);
  pm.addPass(createTileGrid()); // 4c
  addCleanup(pm);
  pm.addPass(createLinalgFoldUnitExtentDimsPass()); // 4d
  addCleanup(pm);
}

void buildTritonToAtlasVirtualPipeline(OpPassManager &pm) {
  pm.addPass(createValidateTTIR()); // 1
  pm.addPass(createNormalize());    // 2
  addCleanup(pm);
  pm.addPass(createConvertTritonComputeToLinalg()); // 3
  // Dtypes before shapes: retyping the untiled kernel touches each op once,
  // and tiling then sees the final types. Pass numbers stay as they were.
  pm.addPass(createLegalizeDtypes()); // 5
  addCleanup(pm);
  buildLegalizeShapesPipeline(pm);                 // 4
  pm.addPass(createAssignSlots());                 // 6
  pm.addPass(createConvertTritonToAtlasVirtual()); // 7
  // Pass 8 (atlas-mlir's verify-atlas-virtual-stream) is not part of this
  // pipeline yet.
}

void buildGluonToAtlasVirtualPipeline(OpPassManager &pm) {
  pm.addPass(createValidateGluon());
  pm.addPass(createLowerGluonLayoutsToTTIR());
  buildTritonToAtlasVirtualPipeline(pm);
}

} // namespace mlir::triton
