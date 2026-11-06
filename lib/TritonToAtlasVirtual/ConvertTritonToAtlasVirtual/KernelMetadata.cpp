// Pass 6's last step: the DRAM bases and ttatlas.kernel on the function. Empty
// for now. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

LogicalResult attachKernelMetadata(ModuleOp) { return success(); }

} // namespace mlir::triton
