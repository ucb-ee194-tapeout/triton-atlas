// Pass 7 patterns for linalg.matmul chains (fill zero + matmuls -> mxu_reset /
// mxu_accumulate / mxu_readout_bf16) and FP8 transposes. Empty for now.
// Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateDotPatterns(TypeConverter &, RewritePatternSet &) {}

} // namespace mlir::triton
