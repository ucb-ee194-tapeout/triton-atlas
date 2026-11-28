// The order of our passes lives here, not in Python, so it can run without
// Triton's stages. Same idea as atlas-mlir's lib/AtlasPipeline.cpp. Order: the
// kernel is validated and normalized, lowered to linalg, retyped, tiled, given
// its DRAM slots, then converted to Atlas virtual IR.
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <cstdlib>

namespace mlir::triton {

// canonicalize + CSE after passes 2, 5 and 4, so dead address math and the
// leftovers of tiling never reach ttatlas-assign-slots. None runs after pass 6:
// cleanup merges equal constants and keeps one slot for them (pass 7 rejects
// that), and pass 7 needs its matmul and pack ops in order.
static void addCleanup(OpPassManager &pm) {
  // The Triton Python wheel owns the pass manager and its canonicalizer.  Our
  // plugin supplies the linalg dialect from a second static library because
  // the wheel does not export it; the two images therefore have distinct
  // linalg trait TypeIDs.  The wheel canonicalizer can call getTerminator()
  // using its TypeID on a valid plugin-created region and abort.  The Atlas
  // passes perform their own local cleanup, so omit this cross-image cleanup
  // only for the Python plugin pipeline (triton-atlas-opt keeps it).
  if (std::getenv("TTATLAS_PLUGIN_RUNTIME")) {
    // The first cleanup is before pass 3 creates any linalg operations, so it
    // is safe and is needed to canonicalize TTIR pointer arithmetic.  All
    // later cleanups would hand linalg regions to the wheel's incompatible
    // trait table.
    static unsigned pluginCleanupCount = 0;
    if (pluginCleanupCount++ != 0)
      return;
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

static void addTraceCheckpoint(OpPassManager &pm) {
  if (std::getenv("TTATLAS_TRACE_PIPELINE"))
    pm.addPass(createPrintOpStatsPass());
}

// Pass 4: pack, fold, tile the block grid, fold unit dimensions, with
// canonicalize and CSE after each step.
void buildLegalizeShapesPipeline(OpPassManager &pm) {
  pm.addPass(createPackBlocks()); // 4a
  addTraceCheckpoint(pm);
  addCleanup(pm);
  pm.addPass(createFoldPackPairs()); // 4b
  addTraceCheckpoint(pm);
  addCleanup(pm);
  pm.addPass(createTileGrid()); // 4c
  addTraceCheckpoint(pm);
  addCleanup(pm);
  // This upstream region rewrite has the same cross-image linalg trait issue
  // as the greedy cleanup when loaded by Triton's Python wheel.  The plugin
  // kernels are already packed/tiled to fixed 32x32 blocks, so retaining a
  // unit grid dimension is harmless and keeps their structured bodies intact.
  if (!std::getenv("TTATLAS_PLUGIN_RUNTIME"))
    pm.addPass(createLinalgFoldUnitExtentDimsPass()); // 4d
  addTraceCheckpoint(pm);
  addCleanup(pm);
}

void buildTritonToAtlasVirtualPipeline(OpPassManager &pm) {
  pm.addPass(createValidateTTIR()); // 1
  addTraceCheckpoint(pm);
  pm.addPass(createNormalize());    // 2
  addTraceCheckpoint(pm);
  addCleanup(pm);
  pm.addPass(createConvertTritonComputeToLinalg()); // 3
  addTraceCheckpoint(pm);
  // Dtypes before shapes: retyping the untiled kernel touches each op once,
  // and tiling then sees the final types. Pass numbers stay as they were.
  pm.addPass(createLegalizeDtypes()); // 5
  addTraceCheckpoint(pm);
  addCleanup(pm);
  buildLegalizeShapesPipeline(pm);                 // 4
  pm.addPass(createAssignSlots());                 // 6
  addTraceCheckpoint(pm);
  pm.addPass(createConvertTritonToAtlasVirtual()); // 7
  addTraceCheckpoint(pm);
  // Pass 8 (atlas-mlir's verify-atlas-virtual-stream) is not part of this
  // pipeline yet.
}

} // namespace mlir::triton
