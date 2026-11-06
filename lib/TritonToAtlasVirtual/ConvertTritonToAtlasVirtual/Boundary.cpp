// Pass 7 patterns for the kernel boundary (the signature; slotted slices of
// tt.load; the slotted insert_slices a tt.store writes). Empty for now. Shape
// follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateBoundaryPatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
