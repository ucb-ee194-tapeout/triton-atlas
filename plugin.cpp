// Entry point of the Atlas plugin: Triton loads this library and calls tritonGetPluginInfo() to learn what it adds
// (triton v3.9.0 include/triton/Tools/PluginUtils.h). Written by hand, as triton-lang/triton-ext@268252b
// backend/AppleGPU/ExportAppleGPU.cpp does (MIT). Our passes get listed here once they exist.
#include "Atlas/AtlasDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "triton/Tools/PluginUtils.h"
#include "triton/Version.h"

#include <iterator>

using namespace mlir::triton;

// Atlas virtual IR lives in func.func and uses arith constants (atlas-mlir examples/handoff/virtual_mlp).
static void registerAtlasDialects(mlir::DialectRegistry *registry) {
  registry->insert<mlir::atlas::AtlasDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  // func promises an inliner interface that lives in a separate extension; without it Triton's inliner pass aborts
  // ("promised by dialect 'func' but never implemented"). The wheel doesn't export it, so we link the library ourselves.
  mlir::func::registerInlinerExtension(*registry);
}

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::DialectInfo dialects[] = {
      {"atlas", "0.1.0", registerAtlasDialects},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAtlasBackend",
      "0.1.0",
      nullptr,
      0, // numPasses
      dialects,
      std::size(dialects),
      nullptr,
      0, // numOps
      TRITON_VERSION,
  };
  return &info;
}
