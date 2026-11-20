// The ttatlas dialect: tags the triton-atlas passes pass to each other, plus
// the shared rejection helper. Every pass includes this header (through
// TritonToAtlasVirtual/Passes.h) and names tags only through the constants
// below.
#ifndef TRITON_ATLAS_TTATLAS_TTATLAS_H
#define TRITON_ATLAS_TTATLAS_TTATLAS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "TTAtlas/TTAtlasEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TTAtlas/TTAtlasAttrs.h.inc"

#include "TTAtlas/TTAtlasDialect.h.inc"

namespace mlir::ttatlas {

// Explicit post-Atlas-lowering handoff; not part of the TTIR passes 1-7.
void registerFinalizeMachineHandoffPass();

// Tag names. Plain tags hold built-in attributes; slot and k_chain hold
// the typed attributes above.
inline constexpr llvm::StringLiteral kKChain =
    "ttatlas.k_chain"; // #ttatlas.k_chain, on tt.dot (pass 2, read by pass 3)
inline constexpr llvm::StringLiteral kSlot =
    "ttatlas.slot"; // #ttatlas.slot, on direct tile loads/stores or boundary
                    // slices (pass 6)
inline constexpr llvm::StringLiteral kPrecisionPolicy =
    "ttatlas.precision_policy"; // string, on the module
inline constexpr llvm::StringLiteral kLayout =
    "ttatlas.layout"; // #ttatlas<layout ...>, on the module
inline constexpr llvm::StringLiteral kDramBase =
    "ttatlas.dram_base"; // i64 input base address, on the module
inline constexpr llvm::StringLiteral kFoldBiasIntoDot =
    "ttatlas.fold_bias_into_dot"; // bool, on the module (backend option; pass
                                  // 2)
inline constexpr llvm::StringLiteral kKernel =
    "ttatlas.kernel"; // JSON string (schema v2), on the function (pass 7)
// Precision policies the passes know. Names are versioned, so changing what a
// policy does means a new name.
inline constexpr llvm::StringLiteral kPrecisionPolicies[] = {"fp8-bf16-v1"};

// Every rejection goes through this, so all passes report the same way and
// tests can match the text: "atlas: unsupported <op-name>: <reason>" at the
// op's source location (the kernel's Python line).
InFlightDiagnostic emitUnsupported(Operation *op, const llvm::Twine &reason);

} // namespace mlir::ttatlas

#endif // TRITON_ATLAS_TTATLAS_TTATLAS_H
