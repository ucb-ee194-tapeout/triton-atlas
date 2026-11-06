// The ttatlas dialect: registration, the typed attributes' verifiers, and the
// check of every ttatlas.* tag on every op. Rules: triton-atlas design agenda,
// "Frozen encodings (v1)". Shape follows triton v3.9.0
// lib/Dialect/TritonGPU/IR/Dialect.cpp.
#include "TTAtlas/TTAtlas.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include "TTAtlas/TTAtlasEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TTAtlas/TTAtlasAttrs.cpp.inc"

#include "TTAtlas/TTAtlasDialect.cpp.inc"

namespace mlir::ttatlas {

void TTAtlasDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TTAtlas/TTAtlasAttrs.cpp.inc"
      >();
}

// A 2-D shape with positive sizes: [rows, cols].
static LogicalResult verifyShape(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape) {
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0)
    return emitError()
           << "logical_shape must be [rows, cols] with positive sizes";
  return success();
}

LogicalResult SlotAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               int32_t index, int32_t arg, SlotRole,
                               AffineMapAttr tileOrigin,
                               ArrayRef<int64_t> logicalShape, Type hostDtype) {
  if (index < 0 || arg < 0)
    return emitError() << "slot index and arg must be non-negative";
  if (!isa<IntegerType, FloatType>(hostDtype))
    return emitError()
           << "slot host_dtype must be a scalar integer or float type";
  AffineMap map = tileOrigin.getValue();
  if (map.getNumResults() != 2 || map.getNumDims() != 0 ||
      map.getNumSymbols() != 6)
    return emitError()
           << "tile_origin must have 2 results, no dims and 6 symbols "
              "(pid_x, pid_y, pid_z, nprog_x, nprog_y, nprog_z)";
  return verifyShape(emitError, logicalShape);
}

LogicalResult KChainAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int32_t id, int32_t pos, int32_t len) {
  if (id < 0 || len < 1 || pos < 0 || pos >= len)
    return emitError() << "k_chain needs id >= 0, len >= 1 and 0 <= pos < len";
  return success();
}

LogicalResult PadAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              PadNeutral, ArrayRef<int64_t> logicalShape) {
  return verifyShape(emitError, logicalShape);
}

// An i32 IntegerAttr in [lo, hi].
static bool isI32In(Attribute value, int64_t lo, int64_t hi) {
  auto integer = dyn_cast<IntegerAttr>(value);
  return integer && integer.getType().isSignlessInteger(32) &&
         integer.getInt() >= lo && integer.getInt() <= hi;
}

// The ttatlas.kernel JSON must parse and say which schema version it follows.
// JSON has one number type, so "version": 1.0 reads as 1 too; that is harmless.
static LogicalResult verifyKernelJson(Operation *op, Attribute value) {
  auto text = dyn_cast<StringAttr>(value);
  if (!text)
    return op->emitOpError() << kKernel << " must be a JSON string";
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text.getValue());
  if (!parsed)
    return op->emitOpError()
           << kKernel
           << " is not valid JSON: " << llvm::toString(parsed.takeError());
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->getInteger("version") != 1)
    return op->emitOpError()
           << kKernel << " must be a JSON object with \"version\": 1";
  return success();
}

// Checks one ttatlas.* tag's value. `op` is where errors are reported; checks
// that compare an op's sibling tags (slot vs host_dtype) live in
// verifyOperationAttribute.
static LogicalResult verifyTag(Operation *op, NamedAttribute attr) {
  StringRef name = attr.getName().getValue();
  Attribute value = attr.getValue();
  auto fail = [&](const Twine &what) {
    return op->emitOpError() << name << " must be " << what;
  };

  if (name == kHostDtype)
    return isa<TypeAttr>(value) ? success() : fail("a type");
  if (name == kMxu)
    return isI32In(value, 0, 1) ? success() : fail("an i32, 0 or 1");
  if (name == kScale)
    return isI32In(value, 122, 128) ? success() : fail("an i32 in 122..128");
  if (name == kSlot)
    return isa<SlotAttr>(value) ? success() : fail("a #ttatlas.slot");
  if (name == kKChain)
    return isa<KChainAttr>(value) ? success() : fail("a #ttatlas.k_chain");
  if (name == kPad)
    return isa<PadAttr>(value) ? success() : fail("a #ttatlas.pad");
  if (name == kLayout)
    return isa<LayoutAttr>(value) ? success() : fail("a #ttatlas<layout ...>");
  if (name == kDramBase) {
    auto base = dyn_cast<IntegerAttr>(value);
    return base && base.getType().isSignlessInteger(64) && base.getInt() >= 0
               ? success()
               : fail("a non-negative i64 address");
  }
  if (name == kPrecisionPolicy) {
    auto policy = dyn_cast<StringAttr>(value);
    if (policy && llvm::is_contained(kPrecisionPolicies, policy.getValue()))
      return success();
    return fail("a known policy name (" +
                llvm::join(std::begin(kPrecisionPolicies),
                           std::end(kPrecisionPolicies), ", ") +
                ")");
  }
  if (name == kKernel)
    return verifyKernelJson(op, value);
  if (name == kLegalized) {
    auto entries = dyn_cast<ArrayAttr>(value);
    auto isEntry = [](Attribute entry) {
      auto dict = dyn_cast<DictionaryAttr>(entry);
      return dict && dict.getAs<StringAttr>("pass") &&
             dict.getAs<StringAttr>("what") && dict.getAs<StringAttr>("loc");
    };
    return entries && llvm::all_of(entries, isEntry)
               ? success()
               : fail("a list of {pass, what, loc} entries");
  }
  return op->emitOpError() << "unknown tag " << name;
}

LogicalResult TTAtlasDialect::verifyOperationAttribute(Operation *op,
                                                       NamedAttribute attr) {
  if (failed(verifyTag(op, attr)))
    return failure();
  // Canonicalize folds and dedups constants by value and rebuilds them without
  // their attributes, so a tag on a constant would silently disappear.
  if (op->hasTrait<OpTrait::ConstantLike>())
    return op->emitOpError()
           << attr.getName().getValue()
           << " is on a constant; canonicalize drops attributes of constants, "
              "so tag the op that uses it";
  // Pass 1's host dtype and pass 5's slot record the same fact; once both exist
  // they must agree. Checked from the slot's side, which sees the sibling tag
  // whatever the attribute order.
  if (auto slot = dyn_cast<SlotAttr>(attr.getValue())) {
    auto hostDtype = op->getAttrOfType<TypeAttr>(kHostDtype);
    if (hostDtype && hostDtype.getValue() != slot.getHostDtype())
      return op->emitOpError()
             << kSlot << " host_dtype disagrees with " << kHostDtype;
  }
  return success();
}

LogicalResult TTAtlasDialect::verifyRegionArgAttribute(Operation *op, unsigned,
                                                       unsigned,
                                                       NamedAttribute attr) {
  return verifyTag(op, attr);
}

LogicalResult TTAtlasDialect::verifyRegionResultAttribute(Operation *op,
                                                          unsigned, unsigned,
                                                          NamedAttribute attr) {
  return verifyTag(op, attr);
}

} // namespace mlir::ttatlas
