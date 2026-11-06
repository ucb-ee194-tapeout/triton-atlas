// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 6 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 6 · ttatlas-assign-slots".
def AssignSlots : Pass<"ttatlas-assign-slots", "mlir::ModuleOp"> {
  let summary = "Give each tile a load reads or a store writes its DRAM slot (#ttatlas.slot)";
  let description = [{
    Out: one slot per distinct (pointer argument, constant tile slice): base and strides from the load's pointer
    math, tile_origin from the slice offset; #ttatlas.slot<index, arg, role, tile_origin, logical_shape, host_dtype>.
    The only pass that numbers slots. Where the tag sits (on tt.load / tt.store, or on their slices) is still open;
    the contract checker expects it on loads and stores until that is settled.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
