// Pass 7 patterns for elementwise linalg ops (linalg.generic over arith/math).
// Empty for now. Shape follows triton v3.9.0
// third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateElementwisePatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
