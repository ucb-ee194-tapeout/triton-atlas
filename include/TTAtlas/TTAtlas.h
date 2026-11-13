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
#include "llvm/ADT/StringRef.h"

#include "TTAtlas/TTAtlasEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TTAtlas/TTAtlasAttrs.h.inc"

#include "TTAtlas/TTAtlasDialect.h.inc"

namespace mlir::ttatlas {

// Tag names. Plain tags hold built-in attributes; slot, k_chain and pad hold
// the typed attributes above.
inline constexpr llvm::StringLiteral kHostDtype =
    "ttatlas.host_dtype"; // TypeAttr, on loads and stores (pass 1)
inline constexpr llvm::StringLiteral kKChain =
    "ttatlas.k_chain"; // #ttatlas.k_chain, on tt.dot (pass 2, read by pass 3)
inline constexpr llvm::StringLiteral kPad =
    "ttatlas.pad"; // #ttatlas.pad, one per padded boundary tile, on its op, not
                   // a constant (pass 4)
inline constexpr llvm::StringLiteral kMaskApply =
    "ttatlas.mask_apply"; // unit, on the linalg.mul / linalg.add that pass 4
                          // puts before a reduction: its constant operand is
                          // a constant mask tile (pass 4, read by 6 and 7)
inline constexpr llvm::StringLiteral kMxu =
    "ttatlas.mxu"; // i32 < target::kMxuUnits, on linalg.matmul (pass 4)
inline constexpr llvm::StringLiteral kScale =
    "ttatlas.scale"; // i32 target::kScaleCode{Min,Max}, on an FP8 pack (pass 5)
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
    "ttatlas.kernel"; // JSON string (schema v1), on the function (pass 7)
// List of {pass, what, loc} entries, one per legalization a pass did (on the
// module). Absent or empty means the kernel was native: it reached Atlas
// without any rewrite. Pass 7 copies it into ttatlas.kernel's
// "legalized" field.
inline constexpr llvm::StringLiteral kLegalized = "ttatlas.legalized";

// Precision policies the passes know. Names are versioned, so changing what a
// policy does means a new name.
inline constexpr llvm::StringLiteral kPrecisionPolicies[] = {"fp8-bf16-v1"};

// Every rejection goes through this, so all passes report the same way and
// tests can match the text: "atlas: unsupported <op-name>: <reason>" at the
// op's source location (the kernel's Python line).
InFlightDiagnostic emitUnsupported(Operation *op, const llvm::Twine &reason);

// Every pass that rewrites something only because the input wasn't in Atlas
// form already (padding a mask, retyping f32, splitting a block) records it
// here, so a kernel is native exactly when ttatlas.legalized stays empty.
// Appends {pass, what, loc ("file:line:col")} to the enclosing module's
// ttatlas.legalized, once per distinct entry. Call it only from a pass that
// runs on the whole module (not one nested on functions, which may run in
// parallel), and only after the rewrite has succeeded: the module attribute
// isn't undone if a conversion rolls back. Not exercised by a test yet; the
// first legalizing pass is its first caller.
void recordLegalization(Operation *op, llvm::StringRef pass,
                        const llvm::Twine &what);

} // namespace mlir::ttatlas

#endif // TRITON_ATLAS_TTATLAS_TTATLAS_H
