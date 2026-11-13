// Pass 7's type conversions (32x32 tensors to !atlas.virtual_* tiles).
#include "Patterns.h"

#include "Atlas/AtlasTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::triton {

void populateTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) -> Type {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor ||
        (tensor.getShape() != ArrayRef<int64_t>{32, 32} &&
         tensor.getShape() != ArrayRef<int64_t>{32}))
      return type;

    Type element = tensor.getElementType();
    // Pass 5 keeps an MXU's float32 accumulator in linalg IR, but its
    // architectural readout and all VPU consumers are BF16 tiles. A row
    // reduction's logical tensor<32xT> result is represented by the same
    // physical BF16 tile: virtual_vpu_reduce fills every row with its value.
    if (element.isBF16() || element.isF32())
      return atlas::VirtualBF16Type::get(type.getContext());
    if (isa<Float8E4M3FNType>(element))
      return atlas::VirtualFP8Type::get(type.getContext());
    return type;
  });
}

} // namespace mlir::triton
