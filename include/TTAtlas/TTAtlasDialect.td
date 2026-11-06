// The ttatlas dialect: the tags our passes pass to each other. It defines attributes only, no ops. Encodings:
// triton-atlas design agenda, "Frozen encodings (v1)". Dialect shape follows triton v3.9.0
// include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td.
#ifndef TRITON_ATLAS_TTATLAS_DIALECT
#define TRITON_ATLAS_TTATLAS_DIALECT

include "mlir/IR/DialectBase.td"

def TTAtlas_Dialect : Dialect {
  let name = "ttatlas";
  let cppNamespace = "::mlir::ttatlas";
  let summary = "Tags the triton-atlas passes pass to each other (attributes only)";
  let description = [{
    Typed attributes (#ttatlas.slot, #ttatlas.k_chain, #ttatlas.pad, #ttatlas<layout ...>) plus plain tags named
    ttatlas.* that hold built-in attributes (ttatlas.mxu = 0 : i32). verifyOperationAttribute (and the region argument and
    result verifiers) check every ttatlas.* name on every op, argument and result, so a misspelled or mistyped tag fails where it is set.
  }];
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;    // tags on function arguments
  let hasRegionResultAttrVerify = 1; // tags on function results
}

#endif // TRITON_ATLAS_TTATLAS_DIALECT
