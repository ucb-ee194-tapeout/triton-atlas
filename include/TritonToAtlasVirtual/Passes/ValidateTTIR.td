// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 1 of the TTIR -> Atlas pipeline. Rejects TTIR no later pass lowers yet.
def ValidateTTIR : Pass<"ttatlas-validate-ttir", "mlir::ModuleOp"> {
  let summary = "Stop TTIR that no later pass lowers yet";
  let description = [{
    Out: one tt.func with only ops later passes handle, in dtypes the later passes accept. What no later pass
    lowers yet is an error naming the op, the reason and the kernel's source line. This pass owns no rules of its
    own: when a later pass learns to lower something, its op is accepted here in the same change, so the set of
    accepted kernels grows over time.

    Known limit: the pass stops at the first op it rejects, so a kernel fixed for that op may fail again on the next.
  }];

  // Loads the ttatlas dialect, so the ops it matches are registered, and every dialect
  // the op inventory names, so its op names are the registered ones.
  let dependentDialects = [
    "mlir::ttatlas::TTAtlasDialect", "mlir::triton::TritonDialect", "mlir::arith::ArithDialect", "mlir::math::MathDialect", "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect", "mlir::ub::UBDialect", "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"
  ];
}
