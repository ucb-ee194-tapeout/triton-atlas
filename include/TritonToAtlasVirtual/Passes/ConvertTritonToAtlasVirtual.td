// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 6 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 6 · convert-triton-to-atlas-virtual".
def ConvertTritonToAtlasVirtual : Pass<"convert-triton-to-atlas-virtual", "mlir::ModuleOp"> {
  let summary = "Convert TTIR to Atlas virtual IR: the compute, the kernel boundary and its metadata";
  let description = [{
    Out: Atlas virtual IR that atlas-mlir accepts. Compute becomes atlas.virtual_* ops on 32x32 tiles, one
    pattern per op or op group. tt.func becomes an argument-free func.func: virtual_start, one virtual_input_* /
    virtual_output_bf16 per slot threaded on the state token, func.return of the final token; casts, loads,
    stores and address math are removed. The function carries atlas.input_dram_base / output_dram_base and
    ttatlas.kernel (JSON, schema v1). An op with no pattern is an error naming it.
  }];

  // The ops this pass creates, and the ttatlas tags it reads; MLIR loads these dialects before the pass runs.
  let dependentDialects = ["mlir::atlas::AtlasDialect", "mlir::func::FuncDialect", "mlir::arith::ArithDialect",
                           "mlir::ttatlas::TTAtlasDialect"];
}
