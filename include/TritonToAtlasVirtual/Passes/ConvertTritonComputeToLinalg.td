// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 3 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 3 · convert-triton-compute-to-linalg".
def ConvertTritonComputeToLinalg : Pass<"convert-triton-compute-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert TTIR compute on tensors to linalg; loads, stores and address math stay TTIR";
  let description = [{
    Out: no tt.dot, tt.trans or tt.reduce, and no arith / math / tt.broadcast / tt.expand_dims producing a float
    tensor: compute is linalg on tensors (elementwise ops through upstream convert-elementwise-to-linalg, each with
    a fresh tensor.empty output; tt.dot -> linalg.fill zero + linalg.matmul; tt.trans -> linalg.transpose; tt.reduce
    -> linalg.reduce; broadcasts -> linalg.broadcast). tt.load / tt.store, pointers and address math stay TTIR, and
    tags are copied onto the new ops. Why linalg: MLIR's tiling then gives the 32x32 tiles in pass 4 (IA-19).
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified. linalg joins
  // with the first pattern, together with linking it into the plugin: libtriton does not export linalg.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
