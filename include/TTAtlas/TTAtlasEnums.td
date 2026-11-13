// Enums used by the ttatlas attributes. Values: triton-atlas design agenda, "Frozen encodings (v1)".
#ifndef TRITON_ATLAS_TTATLAS_ENUMS
#define TRITON_ATLAS_TTATLAS_ENUMS

include "mlir/IR/EnumAttr.td"
include "TTAtlasDialect.td"

// What a slot's tile is for (pass 6 sets it; pass 7 and the driver read it).
def TTAtlas_SlotRole : I32EnumAttr<"SlotRole", "role of a DRAM slot's tile", [
    I32EnumAttrCase<"activation_fp8", 0>, I32EnumAttrCase<"weight_fp8_nk", 1>,
    I32EnumAttrCase<"weight_fp8_kn", 2>, I32EnumAttrCase<"bias_bf16", 3>,
    I32EnumAttrCase<"mask_bf16", 4>, I32EnumAttrCase<"constant_bf16", 5>,
    I32EnumAttrCase<"output_bf16", 6>]> {
  let cppNamespace = "::mlir::ttatlas";
  let genSpecializedAttr = 0;
}

// Neutral element written into padded lanes (pass 4).
def TTAtlas_PadNeutral : I32EnumAttr<"PadNeutral", "neutral element of padded lanes", [
    I32EnumAttrCase<"zero", 0>, I32EnumAttrCase<"neg_inf", 1>,
    I32EnumAttrCase<"one", 2>, I32EnumAttrCase<"pos_inf", 3>]> {
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
