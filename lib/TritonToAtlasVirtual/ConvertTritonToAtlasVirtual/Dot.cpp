// Pass 6 patterns for tt.dot (one dot, trans + dot, K chains). Empty for now.
// Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateDotPatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
