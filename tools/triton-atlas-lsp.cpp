// Language server for .mlir files with Triton, Atlas and every upstream MLIR
// dialect, so one editor window reads both sides of our conversion. A dialect
// has to be compiled into the server:
// https://mlir.llvm.org/docs/Tools/MLIRLSP/. Same shape as triton v3.9.0
// bin/triton-lsp.cpp. make build builds it.
#include "Atlas/AtlasDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::atlas::AtlasDialect, mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect,
                  mlir::triton::gluon::GluonDialect>();
  mlir::registerAllDialects(registry);
  // Both servers have the same dialects; full is built from Triton's source,
  // small is linked to the wheel.
#ifdef TRITON_ATLAS_LSP_FROM_SOURCE
  constexpr const char *kind = "full";
#else
  constexpr const char *kind = "small";
#endif
  // --version says which server this is, so nothing else has to record it.
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << "triton-atlas-lsp " << kind << "\n"; });
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
