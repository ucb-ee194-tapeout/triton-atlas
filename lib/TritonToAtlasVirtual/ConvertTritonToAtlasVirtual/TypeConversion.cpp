// Pass 6's type conversions (TTIR tensors to !atlas.virtual_* tiles). For now
// every type converts to itself, so the conversion runs before any pattern
// exists. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"

namespace mlir::triton {

void populateTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) { return type; });
}

} // namespace mlir::triton
