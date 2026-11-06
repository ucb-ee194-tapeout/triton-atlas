// Pass 6 patterns for tt.reduce + expand_dims + broadcast. Empty for now. Shape
// follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateReducePatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
