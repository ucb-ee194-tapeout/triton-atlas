// The order of our passes lives here, not in Python, so it can run without
// Triton's stages. Same idea as atlas-mlir's lib/AtlasPipeline.cpp. Order: the
// kernel is validated and normalized, lowered to linalg, retyped, tiled, given
// its DRAM slots, then converted to Atlas virtual IR.
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iterator>

namespace mlir::triton {

namespace {

// Triton's Python wheel and this plugin carry separate linalg trait tables.
// The stock fold-unit-extent pass uses a greedy region simplifier that sees
// the plugin's generic body with the wheel's TypeID and clears it.  Keep the
// useful shape rewrite but disable that region simplification in the plugin
// path; the standalone tool continues to use the stock pass.
struct FoldUnitExtentDimsCompat
    : PassWrapper<FoldUnitExtentDimsCompat, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldUnitExtentDimsCompat)
  StringRef getArgument() const final { return "ttatlas-fold-unit-extents"; }
  StringRef getDescription() const final {
    return "fold linalg unit dimensions without cross-image region cleanup";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::ControlDropUnitDims options;
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
    if (std::getenv("TTATLAS_TRACE_OPS")) {
      getOperation()->walk([&](linalg::PackOp pack) {
        llvm::errs() << "ttatlas compat pack result=" << pack.getResult()
                     << " users="
                     << std::distance(pack->getResult(0).use_begin(),
                                      pack->getResult(0).use_end())
                     << "\n";
      });
      getOperation()->walk([&](tensor::CollapseShapeOp collapse) {
        Operation *def = collapse.getSrc().getDefiningOp();
        llvm::errs() << "ttatlas compat collapse src="
                     << (def ? def->getName().getStringRef() : "block-arg")
                     << "\n";
      });
    }
  }
};

} // namespace

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
  if (std::getenv("TTATLAS_PLUGIN_RUNTIME"))
    pm.addPass(std::make_unique<FoldUnitExtentDimsCompat>());
  else
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
