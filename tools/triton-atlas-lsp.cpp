// Language server for .mlir files with Triton, Atlas and MLIR's own ops, so one
// editor window reads both sides of our conversion. A dialect has to be
// compiled into the server: https://mlir.llvm.org/docs/Tools/MLIRLSP/. Same
// shape as triton v3.9.0 bin/triton-lsp.cpp. make build builds it.
#include "Atlas/AtlasDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#ifdef TRITON_ATLAS_LSP_ALL_DIALECTS
#include "mlir/InitAllDialects.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<
      mlir::atlas::AtlasDialect, mlir::triton::TritonDialect,
      mlir::triton::gpu::TritonGPUDialect, mlir::triton::gluon::GluonDialect,
      mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect, mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
      mlir::math::MathDialect, mlir::scf::SCFDialect, mlir::ub::UBDialect>();
#ifdef TRITON_ATLAS_LSP_ALL_DIALECTS
  mlir::registerAllDialects(registry);
  constexpr const char *kind = "full";
#else
  constexpr const char *kind = "small";
#endif
  // --version says which server this is, so nothing else has to record it.
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << "triton-atlas-lsp " << kind << "\n"; });
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
