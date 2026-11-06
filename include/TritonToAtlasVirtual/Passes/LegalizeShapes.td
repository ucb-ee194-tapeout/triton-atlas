// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 3 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 3 · ttatlas-legalize-shapes".
def LegalizeShapes : Pass<"ttatlas-legalize-shapes", "mlir::ModuleOp"> {
  let summary = "Split blocks into 32x32 tiles; tag K chains, padding and MXU units";
  let description = [{
    Out: every 2-D value is exactly 32x32 (a 1:N conversion: a larger value becomes a grid of tile values). Dots
    carry ttatlas.mxu, K > 32 contractions ttatlas.k_chain, padded values ttatlas.pad on their defining ops.
  }];
}
