// Pass 7's type conversions (32x32 tensors to !atlas.virtual_* tiles).
#include "Patterns.h"

#include "Atlas/AtlasTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::triton {

void populateTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) -> Type {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getShape() != ArrayRef<int64_t>{32, 32})
      return type;

    Type element = tensor.getElementType();
    if (element.isBF16())
      return atlas::VirtualBF16Type::get(type.getContext());
    if (isa<Float8E4M3FNType>(element))
      return atlas::VirtualFP8Type::get(type.getContext());
    return type;
  });
}

} // namespace mlir::triton
