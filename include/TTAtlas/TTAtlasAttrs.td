// The typed ttatlas attributes. Changing a field or type needs agreement from
// every pass that reads it. Shape follows triton v3.9.0
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

// On each boundary tile (a linalg.pack tile or a linalg.unpack tile) or direct tt.load / tt.store, set by
// ttatlas-assign-slots (pass 6), read by pass 7 and the host packer.
def TTAtlas_SlotAttr : TTAtlas_Attr<"Slot", "slot"> {
  let summary = "the DRAM slot a tile moves and how the host fills it";
  let description = [{
    One 32x32 tile (or a 32-vector as a block of 32 rows by 1 column, [0, 0, 32, 1]) of device memory.
    index: slot number; inputs and outputs are numbered separately, each from 0. kind: a memory slot is gathered from
    (or scattered to) a pointer argument; a constant slot carries its pattern.
    Memory slots only: arg (the pointer argument), offset (flat element offset of the tile's element (0, 0)),
    row_stride and col_stride (elements). block and valid are [lo_row, lo_col, hi_row, hi_col], tile-local, hi
    exclusive: the host writes memory inside valid, `other` (0 when absent) inside block but outside valid, and 0
    outside block. other: the masked-lane value of the load.
    Constant slots only: pattern, the tile as raw BF16 bits (tensor<32x32xi16>).
    role: what the tile is for. vector_layout: rows or cols for a vector tile, none for a matrix. transposed: the
    host transposes the tile. host_dtype: the pointer's element type; tile_dtype: the tile's.
    Offset, strides, block and valid are plain integers; affine forms in the program IDs come later.
  }];
  let parameters = (ins "int32_t":$index, "::mlir::ttatlas::SlotKind":$kind, "int32_t":$arg,
                        "::mlir::ttatlas::SlotRole":$role, "int64_t":$offset, "int64_t":$row_stride,
                        "int64_t":$col_stride, ArrayRefParameter<"int64_t">:$block,
                        ArrayRefParameter<"int64_t">:$valid, OptionalParameter<"::mlir::FloatAttr">:$other,
                        OptionalParameter<"::mlir::DenseIntElementsAttr">:$pattern,
                        "::mlir::ttatlas::VectorLayout":$vector_layout, "bool":$transposed,
                        "::mlir::Type":$host_dtype, "::mlir::Type":$tile_dtype);
  // The memory-only and constant-only fields are left out of the text of the other kind, and -inf and nan are
  // spelled as words (the generic attribute parser reads only hex for them), so the form is parsed by hand.
  let hasCustomAssemblyFormat = 1;
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

#endif // TRITON_ATLAS_TTATLAS_ATTRS
