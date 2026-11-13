// The pieces of pass 1 (ttatlas-validate-ttir), one file each. Private to this
// folder; ValidateTTIR.cpp runs them.
#ifndef TRITON_ATLAS_VALIDATETTIR_VALIDATETTIR_H
#define TRITON_ATLAS_VALIDATETTIR_VALIDATETTIR_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::validate {

// The error text of each illegal op in the inventory, "<kind>: <detail>".
using Reasons = DenseMap<OperationName, std::string>;

// Why masks (arith.cmpi, arith.andi) are rejected. Pass 4 will turn masks with
// compile-time bounds into padding (pass 4 page, "Grows"); then their rows
// become legal and pass 4 owns the rules on them (runtime bounds, `other`).
inline constexpr StringLiteral kMaskReason =
    "masks; pass 4 doesn't turn them into padding yet";

// Reports why an op breaks a condition. Null when the caller only asks whether
// the condition holds (a legality callback).
using EmitFn = function_ref<InFlightDiagnostic()>;

// The op inventory as a ConversionTarget (OpInventory.cpp): every op TTIR can
// contain is legal, legal under a condition, illegal or dropped. Fills
// `reasons` with each illegal op's error text.
void buildOpInventory(ConversionTarget &target, Reasons &reasons);

// Erases the ops the inventory drops (OpInventory.cpp).
void populateDropPatterns(RewritePatternSet &patterns);

// The condition of a conditionally legal op, e.g. "tt.reduce on axis 1 only"
// (Conditions.cpp). Success for every other op.
LogicalResult checkConditions(Operation *op, EmitFn emit);

// The element type a load or store moves in host memory: the pointee of its
// pointer operand (one pointer or a tensor of them). Triton's type verifier
// rejects a tensor of pointers to pointers ("pointer types must point to
// integer or floating-point types"), so a tensor load or store always has an
// integer or float pointee; a scalar one is rejected as a scalar load or store
// before any host dtype is recorded.
inline Type pointee(Value ptr) {
  return cast<PointerType>(getElementTypeOrSelf(ptr.getType()))
      .getPointeeType();
}

} // namespace mlir::triton::validate

#endif // TRITON_ATLAS_VALIDATETTIR_VALIDATETTIR_H
