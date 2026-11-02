// Runs passes on one .mlir file, for lit tests and debugging: Triton's own
// triton-opt, rebuilt against the stock wheel, which doesn't ship it. It loads
// our plugin the way triton-opt does (triton v3.9.0
// bin/RegisterTritonDialects.h:159-162), so tests run the library we ship.
// TRITON_PLUGIN_PATHS picks the plugin; unset or empty, the one this build
// made.
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"

#include <cstdlib>

int main(int argc, char **argv) {
  const char *paths = std::getenv("TRITON_PLUGIN_PATHS");
  if (!paths || !*paths)
    setenv("TRITON_PLUGIN_PATHS", TRITON_ATLAS_PLUGIN, /*overwrite=*/1);
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::triton::gluon::GluonDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses(); // make_ttir's passes
  for (const auto &plugin : mlir::triton::plugin::loadPlugins()) {
    plugin.registerPasses();
    plugin.registerDialects(registry);
  }
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "triton-atlas-opt\n", registry));
}
