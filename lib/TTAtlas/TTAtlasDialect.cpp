// The ttatlas dialect: registration, the typed attributes' verifiers, and the
// check of every ttatlas.* tag on every op. Shape follows triton v3.9.0
// lib/Dialect/TritonGPU/IR/Dialect.cpp.
#include "TTAtlas/AtlasTarget.h"
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

// A rectangle [lo_row, lo_col, hi_row, hi_col] inside the 32x32 tile.
static bool isTileBox(ArrayRef<int64_t> box, bool mayBeEmpty) {
  if (box.size() != 4)
    return false;
  for (int axis = 0; axis < 2; ++axis)
    if (box[axis] < 0 || box[axis + 2] > 32 ||
        (mayBeEmpty ? box[axis] > box[axis + 2] : box[axis] >= box[axis + 2]))
      return false;
  return true;
}

LogicalResult SlotAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               int32_t index, SlotKind kind, int32_t arg,
                               SlotRole, int64_t offset, int64_t rowStride,
                               int64_t colStride, ArrayRef<int64_t> block,
                               ArrayRef<int64_t> valid, FloatAttr other,
                               DenseIntElementsAttr pattern, VectorLayout, bool,
                               Type hostDtype, Type tileDtype) {
  if (index < 0 || arg < 0 || offset < 0)
    return emitError() << "slot index, arg and offset must be non-negative";
  if (!isa<IntegerType, FloatType>(hostDtype) ||
      !isa<IntegerType, FloatType>(tileDtype))
    return emitError() << "slot host_dtype and tile_dtype must be scalar "
                          "integer or float types";
  if (!isTileBox(block, false) || !isTileBox(valid, true))
    return emitError() << "slot block must be a non-empty and valid a "
                          "possibly empty [lo_row, lo_col, hi_row, hi_col] "
                          "box inside the 32x32 tile";
  for (int axis = 0; axis < 2; ++axis)
    if (valid[axis] < block[axis] || valid[axis + 2] > block[axis + 2])
      return emitError() << "slot valid box must lie inside its block";
  if (kind == SlotKind::memory) {
    if (other && other.getType() != tileDtype && other.getType() != hostDtype)
      return emitError() << "slot other must have the host or tile dtype";
    return success();
  }
  if (arg != 0 || offset != 0 || rowStride != 0 || colStride != 0 || other)
    return emitError()
           << "a constant slot has no arg, offset, strides or other";
  auto type = pattern ? dyn_cast<RankedTensorType>(pattern.getType())
                      : RankedTensorType();
  if (!type || type.getShape() != ArrayRef<int64_t>{32, 32} ||
      !type.getElementType().isInteger(16))
    return emitError()
           << "a constant slot needs a pattern of type tensor<32x32xi16>";
  return success();
}

// Parses `name =`.
static ParseResult parseField(AsmParser &parser, StringRef name) {
  return failure(parser.parseKeyword(name) || parser.parseEqual());
}

template <typename Enum>
static ParseResult parseEnumField(AsmParser &parser, StringRef name,
                                  std::optional<Enum> (*symbolize)(StringRef),
                                  Enum &result) {
  StringRef keyword;
  SMLoc loc = parser.getCurrentLocation();
  if (parseField(parser, name) || parser.parseKeyword(&keyword))
    return failure();
  auto value = symbolize(keyword);
  if (!value)
    return parser.emitError(loc, "unknown ") << name << " '" << keyword << "'";
  result = *value;
  return success();
}

static ParseResult parseIntField(AsmParser &parser, StringRef name,
                                 int64_t &result) {
  return failure(parseField(parser, name) || parser.parseInteger(result));
}

static ParseResult parseBox(AsmParser &parser, StringRef name,
                            SmallVectorImpl<int64_t> &box) {
  return failure(
      parseField(parser, name) ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        box.emplace_back();
        return parser.parseInteger(box.back());
      }));
}

// `inf`, `-inf` and `nan` are words here; the generic float parser reads only
// hex for them.
static ParseResult parseOther(AsmParser &parser, FloatAttr &other) {
  bool negative = succeeded(parser.parseOptionalMinus());
  bool inf = succeeded(parser.parseOptionalKeyword("inf"));
  bool nan = !inf && !negative && succeeded(parser.parseOptionalKeyword("nan"));
  if (!inf && !nan) {
    if (negative)
      return parser.emitError(parser.getCurrentLocation(), "expected inf");
    return parser.parseAttribute(other);
  }
  FloatType type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();
  const llvm::fltSemantics &semantics = type.getFloatSemantics();
  other = FloatAttr::get(type, inf ? APFloat::getInf(semantics, negative)
                                   : APFloat::getQNaN(semantics));
  return success();
}

Attribute SlotAttr::parse(AsmParser &parser, Type) {
  int64_t index = 0, arg = 0, offset = 0, rowStride = 0, colStride = 0;
  SlotKind kind;
  SlotRole role;
  VectorLayout layout;
  SmallVector<int64_t, 4> block, valid;
  FloatAttr other;
  DenseIntElementsAttr pattern;
  Type hostDtype, tileDtype;
  StringRef transposed;
  auto comma = [&] { return parser.parseComma(); };
  if (parser.parseLess() || parseIntField(parser, "index", index) || comma() ||
      parseEnumField(parser, "kind", symbolizeSlotKind, kind) || comma())
    return {};
  if (kind == SlotKind::memory &&
      (parseIntField(parser, "arg", arg) || comma()))
    return {};
  if (parseEnumField(parser, "role", symbolizeSlotRole, role) || comma())
    return {};
  if (kind == SlotKind::memory &&
      (parseIntField(parser, "offset", offset) || comma() ||
       parseIntField(parser, "row_stride", rowStride) || comma() ||
       parseIntField(parser, "col_stride", colStride) || comma()))
    return {};
  if (parseBox(parser, "block", block) || comma() ||
      parseBox(parser, "valid", valid) || comma())
    return {};
  if (kind == SlotKind::constant) {
    if (parseField(parser, "pattern") || parser.parseAttribute(pattern) ||
        comma())
      return {};
  } else if (succeeded(parser.parseOptionalKeyword("other"))) {
    if (parser.parseEqual() || parseOther(parser, other) || comma())
      return {};
  }
  SMLoc transposedLoc;
  if (parseEnumField(parser, "vector_layout", symbolizeVectorLayout, layout) ||
      comma() || parseField(parser, "transposed") ||
      (transposedLoc = parser.getCurrentLocation(),
       parser.parseKeyword(&transposed)) ||
      comma() || parseField(parser, "host_dtype") ||
      parser.parseType(hostDtype) || comma() ||
      parseField(parser, "tile_dtype") || parser.parseType(tileDtype) ||
      parser.parseGreater())
    return {};
  if (transposed != "true" && transposed != "false") {
    parser.emitError(transposedLoc, "transposed must be true or false");
    return {};
  }
  return parser.getChecked<SlotAttr>(
      parser.getContext(), static_cast<int32_t>(index), kind,
      static_cast<int32_t>(arg), role, offset, rowStride, colStride, block,
      valid, other, pattern, layout, transposed == "true", hostDtype,
      tileDtype);
}

void SlotAttr::print(AsmPrinter &printer) const {
  bool memory = getKind() == SlotKind::memory;
  printer << "<index = " << getIndex()
          << ", kind = " << stringifySlotKind(getKind());
  if (memory)
    printer << ", arg = " << getArg();
  printer << ", role = " << stringifySlotRole(getRole());
  if (memory)
    printer << ", offset = " << getOffset()
            << ", row_stride = " << getRowStride()
            << ", col_stride = " << getColStride();
  printer << ", block = [" << getBlock() << "], valid = [" << getValid() << "]";
  if (FloatAttr other = getOther()) {
    APFloat value = other.getValue();
    printer << ", other = ";
    if (value.isNaN())
      printer << "nan : " << other.getType();
    else if (value.isInfinity())
      printer << (value.isNegative() ? "-inf" : "inf") << " : "
              << other.getType();
    else
      printer << other;
  }
  if (DenseIntElementsAttr pattern = getPattern())
    printer << ", pattern = " << pattern;
  printer << ", vector_layout = " << stringifyVectorLayout(getVectorLayout())
          << ", transposed = " << (getTransposed() ? "true" : "false")
          << ", host_dtype = " << getHostDtype()
          << ", tile_dtype = " << getTileDtype() << ">";
}

LogicalResult KChainAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int32_t id, int32_t pos, int32_t len) {
  if (id < 0 || len < 1 || pos < 0 || pos >= len)
    return emitError() << "k_chain needs id >= 0, len >= 1 and 0 <= pos < len";
  return success();
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
  if (!object || object->getInteger("version") != 2)
    return op->emitOpError()
           << kKernel << " must be a JSON object with \"version\": 2";
  return success();
}

// Checks one ttatlas.* tag's value. `op` is where errors are reported.
static LogicalResult verifyTag(Operation *op, NamedAttribute attr) {
  StringRef name = attr.getName().getValue();
  Attribute value = attr.getValue();
  auto fail = [&](const Twine &what) {
    return op->emitOpError() << name << " must be " << what;
  };

  if (name == kSlot) {
    auto isSlot = [](Attribute slot) { return isa<SlotAttr>(slot); };
    auto list = dyn_cast<ArrayAttr>(value);
    return isSlot(value) || (list && llvm::all_of(list, isSlot))
               ? success()
               : fail("a #ttatlas.slot or a list of them");
  }
  if (name == kKChain)
    return isa<KChainAttr>(value) ? success() : fail("a #ttatlas.k_chain");
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
  if (name == kFoldBiasIntoDot)
    return isa<BoolAttr>(value) ? success() : fail("a bool");
  if (name == kKernel)
    return verifyKernelJson(op, value);
  return op->emitOpError() << "unknown tag " << name;
}

LogicalResult TTAtlasDialect::verifyOperationAttribute(Operation *op,
                                                       NamedAttribute attr) {
  if (failed(verifyTag(op, attr)))
    return failure();
  // Canonicalize folds and dedups constants by value and rebuilds them without
  // their attributes, so a tag on a constant would silently disappear.
  if (op->hasTrait<OpTrait::ConstantLike>() && attr.getName() != kSlot)
    return op->emitOpError()
           << attr.getName().getValue()
           << " is on a constant; canonicalize drops attributes of constants, "
              "so tag the op that uses it";
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
