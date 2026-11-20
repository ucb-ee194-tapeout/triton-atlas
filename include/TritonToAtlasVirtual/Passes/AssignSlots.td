// Included from ../Passes.td, which includes mlir/Pass/PassBase.td first.
// Pass 6 of the TTIR -> Atlas pipeline. Assigns DRAM slots to tile loads and stores.
def AssignSlots : Pass<"ttatlas-assign-slots", "mlir::ModuleOp"> {
  let summary = "Give each tile a load reads or a store writes its DRAM slot (#ttatlas.slot)";
  let description = [{
    Out: one #ttatlas.slot (schema in TTAtlasAttrs.td) per tile of the memory boundary pass 4 leaves, as an array
    in row-major grid order on each linalg.pack of a tt.load and each linalg.unpack that a tt.store writes (the
    bare slot for a single tile). A kernel with no compute gets its slot on the tt.load / tt.store itself, and
    non-splat BF16 constants (the clamp tiles) get constant slots. Offset and strides come from the pointer
    math, valid from the mask, other from a masked load's constant other. The only pass that numbers slots.
    Rejects what one slot cannot describe: a pointer that is not a kernel argument plus compile-time offsets
    (including one that depends on a program id), a non-constant or negative stride, a mask that is not one
    compile-time rectangle, a masked load whose other is not a constant, overlapping stores, a load of an
    earlier store, and FP8 stores.
  }];

  // Loads the ttatlas dialect, so the tags this pass reads and writes can be built and are verified.
  let dependentDialects = ["mlir::ttatlas::TTAtlasDialect"];
}
