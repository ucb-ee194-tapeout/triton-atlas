// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 1 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 1 · ttatlas-validate-ttir".
def ValidateTTIR : Pass<"ttatlas-validate-ttir", "mlir::ModuleOp"> {
  let summary = "Stop TTIR that no later pass lowers yet; record each load/store's host dtype";
  let description = [{
    Out: one tt.func with only ops later passes handle, in dtypes the precision policy accepts. Every tt.load and
    tt.store carries ttatlas.host_dtype, recorded before ttatlas-legalize-dtypes changes types. What no later pass
    lowers yet is an error naming the op, the reason and the kernel's source line. This pass owns no rules of its
    own: when a later pass learns to lower something, its op is accepted here in the same change, so the set of
    accepted kernels grows over time.

    Known limit: the pass stops at the first op it rejects, so a kernel fixed for that op may fail again on the next.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified, and every dialect
  // the op inventory names, so its op names are the registered ones.
  let dependentDialects = [
    "mlir::ttatlas::TTAtlasDialect", "mlir::triton::TritonDialect", "mlir::arith::ArithDialect", "mlir::math::MathDialect", "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect", "mlir::ub::UBDialect", "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"
  ];
}
