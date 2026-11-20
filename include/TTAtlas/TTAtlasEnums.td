// Enums used by the ttatlas attributes.
#ifndef TRITON_ATLAS_TTATLAS_ENUMS
#define TRITON_ATLAS_TTATLAS_ENUMS

include "mlir/IR/EnumAttr.td"
include "TTAtlasDialect.td"

// What a slot's tile is for (pass 6 sets it; pass 7 and the driver read it).
// A weight's role names the HOST layout of its tile: weight_fp8_nk is [n][k], weight_fp8_kn is [k][n]. The tile the
// matmul reads is always [n][k] (the MXU's form); when the host layout is kn the slot says transposed = true and the
// packer writes the tile turned, so pass 7 adds no XLU transpose for it. Pass 7's XLU transpose is only for a matmul
// whose own operand map reads [k][n], which pass 4 never produces from a packed weight.
def TTAtlas_SlotRole : I32EnumAttr<"SlotRole", "role of a DRAM slot's tile", [
    I32EnumAttrCase<"activation_fp8", 0>, I32EnumAttrCase<"weight_fp8_nk", 1>,
    I32EnumAttrCase<"weight_fp8_kn", 2>, I32EnumAttrCase<"bias_bf16", 3>,
    I32EnumAttrCase<"mask_bf16", 4>, I32EnumAttrCase<"constant_bf16", 5>,
    I32EnumAttrCase<"output_bf16", 6>]> {
  let cppNamespace = "::mlir::ttatlas";
  let genSpecializedAttr = 0;
}

// A memory slot is gathered from a pointer argument; a constant slot carries its own pattern.
def TTAtlas_SlotKind : I32EnumAttr<"SlotKind", "kind of a DRAM slot", [
    I32EnumAttrCase<"memory", 0>, I32EnumAttrCase<"constant", 1>]> {
  let cppNamespace = "::mlir::ttatlas";
  let genSpecializedAttr = 0;
}

// How a vector tile is spread over the 32x32 tile (none: not a vector).
def TTAtlas_VectorLayout : I32EnumAttr<"VectorLayout", "layout of a vector tile", [
    I32EnumAttrCase<"rows", 0>, I32EnumAttrCase<"cols", 1>, I32EnumAttrCase<"none", 2>]> {
  let cppNamespace = "::mlir::ttatlas";
  let genSpecializedAttr = 0;
}

// What the driver promises about DRAM (a backend option).
def TTAtlas_Layout : I32EnumAttr<"Layout", "DRAM layout of host tensors", [
    I32EnumAttrCase<"row_major", 0>, I32EnumAttrCase<"tile_major", 1>]> {
  let cppNamespace = "::mlir::ttatlas";
  let genSpecializedAttr = 0;
}

// The module tag ttatlas.layout holds this attribute, printed #ttatlas<layout tile_major>.
def TTAtlas_LayoutAttr : EnumAttr<TTAtlas_Dialect, TTAtlas_Layout, "layout">;

#endif // TRITON_ATLAS_TTATLAS_ENUMS
