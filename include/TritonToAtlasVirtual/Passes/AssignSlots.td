// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 6 of the TTIR -> Atlas pipeline. Assigns DRAM slots to tile loads and stores.
def AssignSlots : Pass<"ttatlas-assign-slots", "mlir::ModuleOp"> {
  let summary = "Give each tile a load reads or a store writes its DRAM slot (#ttatlas.slot)";
  let description = [{
    Out: one slot per distinct (pointer argument, constant tile slice): base and strides from the load's pointer
    math, offset and strides from the slice offset; the #ttatlas.slot schema is in TTAtlasAttrs.td.
    The only pass that numbers slots. Direct 32x32 accesses carry the tag on tt.load / tt.store.
    Larger loads carry a tag on each constant 32x32 tensor.extract_slice; larger stores carry a tag
    on each tensor.insert_slice in a disjoint, complete assembly of a fresh tensor.empty.
    Slice origins include the parent pointer offset. Masks and dynamic or non-row-major accesses are rejected.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
