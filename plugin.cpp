// Entry point of the Atlas plugin: Triton loads this library and calls
// tritonGetPluginInfo() to learn what it adds (triton v3.9.0
// include/triton/Tools/PluginUtils.h). Written by hand, as
// triton-lang/triton-ext@268252b backend/AppleGPU/ExportAppleGPU.cpp does
// (MIT).
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "TTAtlas/TTAtlas.h"
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Tools/PluginUtils.h"
#include "triton/Version.h"

#include <iterator>

using namespace mlir::triton;

// Atlas virtual IR lives in func.func and uses arith constants (atlas-mlir
// examples/handoff/virtual_mlp). ttatlas holds the tags our passes pass to each
// other.
static void registerAtlasDialects(mlir::DialectRegistry *registry) {
  registry->insert<mlir::atlas::AtlasDialect, mlir::ttatlas::TTAtlasDialect,
                   mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  // func promises an inliner interface that lives in a separate extension;
  // without it Triton's inliner pass aborts
  // ("promised by dialect 'func' but never implemented"). The wheel doesn't
  // export it, so we link the library ourselves.
  mlir::func::registerInlinerExtension(*registry);
  // linalg and tensor ops implement MLIR's TilingInterface through external
  // models that a registry must add (triton-atlas-opt gets them from
  // registerAllExtensions). Without them, pass 4 can't tile or pad anything
  // inside Triton's own process, so a kernel larger than one tile fails there.
  mlir::linalg::registerTilingInterfaceExternalModels(*registry);
  mlir::tensor::registerTilingInterfaceExternalModels(*registry);
}

// Our passes, plus atlas-mlir's check that its virtual IR is well formed, so
// tests and triton-atlas-opt can run --verify-atlas-virtual-stream on our
// output.
static void registerAtlasPasses() {
  registerTritonToAtlasVirtualPasses();
  mlir::ttatlas::registerFinalizeMachineHandoffPass();
  mlir::atlas::registerVerifyAtlasVirtualStreamPass();
  // The whole pipeline under one name, so tests (triton-atlas-opt
  // --ttatlas-pipeline) run exactly what Pipeline.cpp builds instead of a
  // copied pass list.
  static mlir::PassPipelineRegistration<> pipeline(
      "ttatlas-pipeline", "Run the TTIR -> Atlas virtual passes in order",
      [](mlir::OpPassManager &pm) { buildTritonToAtlasVirtualPipeline(pm); });
}

// Python sees this as passes.plugin.add_atlas_virtual_pipeline(pm). The
// callback gets the whole PassManager, so one call adds our full pipeline.
static void addAtlasVirtualPipeline(mlir::PassManager *pm,
                                    const std::vector<std::string> &) {
  buildTritonToAtlasVirtualPipeline(*pm);
}

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"atlas_virtual_pipeline", "0.1.0", addAtlasVirtualPipeline,
       registerAtlasPasses},
  };

  static plugin::DialectInfo dialects[] = {
      {"atlas", "0.1.0", registerAtlasDialects},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAtlasBackend",
      "0.1.0",
      passes,
      std::size(passes),
      dialects,
      std::size(dialects),
      nullptr,
      0, // numOps
      TRITON_VERSION,
  };
  return &info;
}
