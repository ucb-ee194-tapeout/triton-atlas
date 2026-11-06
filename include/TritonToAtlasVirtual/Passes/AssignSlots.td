// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 5 of the TTIR -> Atlas pipeline. Contract: triton-atlas design agenda, page "Pass 5 · ttatlas-assign-slots".
def AssignSlots : Pass<"ttatlas-assign-slots", "mlir::ModuleOp"> {
  let summary = "Tag each tt.load / tt.store with its DRAM slot (#ttatlas.slot)";
  let description = [{
    Out: every load and store carries #ttatlas.slot<index, arg, role, tile_origin, logical_shape, host_dtype>, proven
    to be one whole tile (or a 1-D bias) of one pointer argument. The only pass that numbers slots.
  }];
}
