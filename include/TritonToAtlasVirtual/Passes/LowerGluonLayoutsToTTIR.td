// G1: erase layout-only Gluon types before the shared TTIR pipeline.
def LowerGluonLayoutsToTTIR : Pass<"ttatlas-lower-gluon-layouts-to-ttir", "mlir::ModuleOp"> {
  let summary = "Erase validated Gluon tensor layouts into layout-free TTIR";
  let description = [{
    Converts statically shaped ranked tensor types with a Gluon or TritonGPU
    encoding to the same logical tensor type without an encoding.  A
    gluon.set_auto_layout is erased once both sides have that type.  The first
    implementation intentionally accepts region-free dataflow only; control
    flow and reductions need explicit region/block-argument conversion rules.
  }];
  let dependentDialects = [
    "mlir::triton::gluon::GluonDialect",
    "mlir::triton::TritonDialect"
  ];
}
