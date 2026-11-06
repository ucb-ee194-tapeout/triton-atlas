// Pass 6 patterns for the kernel boundary (signature, slotted loads and
// stores). Empty for now. Shape follows triton v3.9.0
// third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateBoundaryPatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
