// The typed ttatlas attributes. Fields and types: triton-atlas design agenda,
// "Frozen encodings (v1)"; changing one needs agreement from every pass that
// reads it. Shape follows triton v3.9.0
// include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td.
#ifndef TRITON_ATLAS_TTATLAS_ATTRS
#define TRITON_ATLAS_TTATLAS_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "TTAtlasDialect.td"
include "TTAtlasEnums.td"

// Every typed attribute verifies its fields; build them with getChecked() when the values come from input.
class TTAtlas_Attr<string name, string attrMnemonic> : AttrDef<TTAtlas_Dialect, name> {
  let mnemonic = attrMnemonic;
  let genVerifyDecl = 1;
}

// On each direct tile tt.load / tt.store or boundary extract_slice /
// insert_slice, set by ttatlas-assign-slots (pass 6), read by pass 7.
def TTAtlas_SlotAttr : TTAtlas_Attr<"Slot", "slot"> {
  let summary = "the DRAM slot a load or store moves, and where its tile sits in the host tensor";
  let description = [{
    index: slot number; inputs and outputs are numbered separately, each from 0 (address = base + index * 2048).
    arg: index of the pointer argument it reads or writes. role: what the tile is for. tile_origin: the tile's row
    and column in the host tensor, in elements: an affine map with 2 results, no dims and the 6 symbols
    (pid_x, pid_y, pid_z, nprog_x, nprog_y, nprog_z). logical_shape: [rows, cols] before padding. host_dtype: the
    host tensor's element type.
  }];
  let parameters = (ins "int32_t":$index, "int32_t":$arg, "::mlir::ttatlas::SlotRole":$role,
                        "::mlir::AffineMapAttr":$tile_origin, ArrayRefParameter<"int64_t">:$logical_shape,
                        "::mlir::Type":$host_dtype);
  // Spelled out instead of struct(params): the shape's comma list needs brackets, or the parser can't tell where it
  // ends and the next field begins.
  let assemblyFormat = "`<` `index` `=` $index `,` `arg` `=` $arg `,` `role` `=` $role `,` `tile_origin` `=` "
                       "$tile_origin `,` `logical_shape` `=` `[` $logical_shape `]` `,` `host_dtype` `=` $host_dtype `>`";
}

// On each tt.dot of an unrolled K loop, set by pass 2, read by pass 3. From pass 3 on, a K chain is the def-use
// chain linalg.fill -> linalg.matmul -> linalg.matmul ... itself, so passes 4-7 read the structure, not this tag.
def TTAtlas_KChainAttr : TTAtlas_Attr<"KChain", "k_chain"> {
  let summary = "position of a dot in one K contraction";
  let description = [{
    id: one per contraction in the kernel. pos: 0-based step. len: steps in the chain. Pass 3 links the tagged
    dots into one linalg.fill -> linalg.matmul chain; pass 7 emits mxu_reset at its start and mxu_readout_bf16 at
    its end.
  }];
  let parameters = (ins "int32_t":$id, "int32_t":$pos, "int32_t":$len);
  let assemblyFormat = "`<` struct(params) `>`";
}

// On the op defining a padded value, set by ttatlas-legalize-shapes (pass 4), read by passes 6 and 7.
def TTAtlas_PadAttr : TTAtlas_Attr<"Pad", "pad"> {
  let summary = "neutral element of a value's padded lanes and its unpadded shape";
  let parameters = (ins "::mlir::ttatlas::PadNeutral":$neutral, ArrayRefParameter<"int64_t">:$logical_shape);
  let assemblyFormat = "`<` `neutral` `=` $neutral `,` `logical_shape` `=` `[` $logical_shape `]` `>`";
}

#endif // TRITON_ATLAS_TTATLAS_ATTRS
