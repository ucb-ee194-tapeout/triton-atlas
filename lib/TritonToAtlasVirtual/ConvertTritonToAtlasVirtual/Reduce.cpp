// Pass 7 patterns for linalg.reduce (+ its broadcast back to 32x32). Empty for
// now. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateReducePatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
