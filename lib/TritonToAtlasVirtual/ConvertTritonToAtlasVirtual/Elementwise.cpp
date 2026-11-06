// Pass 6 patterns for elementwise arith/math ops. Empty for now. Shape follows
// triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateElementwisePatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
