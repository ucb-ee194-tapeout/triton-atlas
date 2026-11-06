// First native TTIR -> Atlas virtual lowering: one static two-layer FP8 MLP.
// The admitted pattern and precision boundaries match the reference mapper in
// ttir2linalg's amy/triton-atlas-compiler branch. Unsupported kernels fail
// closed; this pass does not erase operations it has not understood.
#include "TritonToAtlasVirtual/Passes.h"

#include "Atlas/AtlasOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONTOATLASVIRTUAL
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

using llvm::SmallPtrSet;
using llvm::SmallVector;
using llvm::StringRef;

static StringRef ttirOperationName(Operation *op) {
  return op->getName().getStringRef();
}

static bool hasTensorShape(Value value, int64_t rows, int64_t cols) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 2 && type.getDimSize(0) == rows &&
         type.getDimSize(1) == cols;
}

static bool typeTextContains(Type type, StringRef needle) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return StringRef(text).contains(needle);
}

static bool isFp8Tile(Value value) {
  return hasTensorShape(value, 32, 32) &&
         typeTextContains(value.getType(), "f8E4M3FN");
}

static bool isBf16Tile(Value value) {
  return hasTensorShape(value, 32, 32) &&
         typeTextContains(value.getType(), "bf16");
}

static bool isF32Tile(Value value) {
  return hasTensorShape(value, 32, 32) &&
         typeTextContains(value.getType(), "xf32");
}

static bool isZeroTensor(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || ttirOperationName(definition) != "arith.constant")
    return false;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(definition->getAttr("value"));
  if (!dense || !dense.isSplat())
    return false;
  return dense.getSplatValue<llvm::APFloat>().isZero();
}

// Follow only the address-forming operations emitted by this 32x32 Triton
// kernel. Returning nullopt rejects indirect/ambiguous memory accesses.
static std::optional<unsigned> pointerArgument(
    Value value, llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getArgNumber();
  Operation *definition = value.getDefiningOp();
  if (!definition || !seen.insert(definition).second ||
      definition->getNumOperands() == 0)
    return std::nullopt;
  StringRef name = ttirOperationName(definition);
  if (name != "tt.splat" && name != "tt.addptr" && name != "tt.broadcast" &&
      name != "tt.expand_dims")
    return std::nullopt;
  return pointerArgument(definition->getOperand(0), seen);
}

static std::optional<unsigned> pointerArgument(Value value) {
  SmallPtrSet<Operation *, 16> seen;
  return pointerArgument(value, seen);
}

static bool dependsOn(Value value, Value source, unsigned depth = 0) {
  if (value == source)
    return true;
  if (depth > 16)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringRef name = ttirOperationName(definition);
  if (name != "arith.truncf" && name != "arith.extf" &&
      name != "tt.fp_to_fp" && name != "arith.maxnumf")
    return false;
  for (Value operand : definition->getOperands())
    if (dependsOn(operand, source, depth + 1))
      return true;
  return false;
}

static bool hasZeroReluPath(Value value, Value source, unsigned depth = 0) {
  if (depth > 16)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringRef name = ttirOperationName(definition);
  if (name == "arith.maxnumf" && definition->getNumOperands() == 2) {
    return (isZeroTensor(definition->getOperand(0)) &&
            dependsOn(definition->getOperand(1), source)) ||
           (isZeroTensor(definition->getOperand(1)) &&
            dependsOn(definition->getOperand(0), source));
  }
  if (name != "arith.truncf" && name != "arith.extf" &&
      name != "tt.fp_to_fp")
    return false;
  for (Value operand : definition->getOperands())
    if (hasZeroReluPath(operand, source, depth + 1))
      return true;
  return false;
}

static bool hasI32EnumAttribute(Operation *op, StringRef name,
                                int64_t expectedValue) {
  auto value = dyn_cast_or_null<IntegerAttr>(op->getAttr(name));
  return value && value.getInt() == expectedValue;
}

static bool isAcceptedDot(Operation *dot) {
  if (ttirOperationName(dot) != "tt.dot" || dot->getNumOperands() != 3 ||
      dot->getNumResults() != 1 || !isFp8Tile(dot->getOperand(0)) ||
      !isFp8Tile(dot->getOperand(1)) || !isZeroTensor(dot->getOperand(2)) ||
      !isF32Tile(dot->getResult(0)))
    return false;

  // AtlasOptions defaults to IEEE for this backend. An explicit non-IEEE
  // request is not admitted by this lowering.
  // Triton 3.9.0 stores these I32EnumAttrs numerically: IEEE=2.
  if (Attribute precision = dot->getAttr("inputPrecision")) {
    auto value = dyn_cast<IntegerAttr>(precision);
    if (!value || value.getInt() != 2)
      return false;
  }
  return true;
}

static bool onlyExpectedOps(Operation *function) {
  static constexpr std::array<StringRef, 16> allowed = {
      "tt.func",       "tt.return",      "tt.make_range",
      "tt.expand_dims", "arith.constant", "arith.muli",
      "tt.splat",      "tt.addptr",      "tt.broadcast",
      "tt.load",       "tt.dot",         "arith.truncf",
      "arith.extf",    "arith.maxnumf",  "tt.fp_to_fp",
      "tt.store"};
  bool accepted = true;
  function->walk([&](Operation *nested) {
    if (nested == function)
      return;
    StringRef name = ttirOperationName(nested);
    if (llvm::none_of(allowed, [&](StringRef candidate) {
          return candidate == name;
        }))
      accepted = false;
  });
  return accepted;
}

struct ConvertTritonToAtlasVirtual
    : impl::ConvertTritonToAtlasVirtualBase<ConvertTritonToAtlasVirtual> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Operation *ttFunc = nullptr;
    for (Operation &op : module.getBody()->getOperations()) {
      if (ttirOperationName(&op) == "tt.func") {
        if (ttFunc) {
          module.emitError("first Atlas lowering expects exactly one tt.func");
          signalPassFailure();
          return;
        }
        ttFunc = &op;
      }
    }
    if (!ttFunc) {
      module.emitError("no tt.func found for Atlas lowering");
      signalPassFailure();
      return;
    }
    if (!onlyExpectedOps(ttFunc)) {
      ttFunc->emitError("unsupported operation in the initial two-layer MLP lowering");
      signalPassFailure();
      return;
    }
    if (ttFunc->getNumRegions() != 1 || ttFunc->getRegion(0).empty()) {
      ttFunc->emitError("expected one nonempty tt.func body");
      signalPassFailure();
      return;
    }

    Block &sourceBlock = ttFunc->getRegion(0).front();
    if (sourceBlock.getNumArguments() != 4) {
      ttFunc->emitError("expected X, W0, W1, and Out pointer arguments");
      signalPassFailure();
      return;
    }

    std::array<Operation *, 4> loads = {nullptr, nullptr, nullptr, nullptr};
    Operation *store = nullptr;
    SmallVector<Operation *, 2> dots;
    for (Operation &nested : sourceBlock.getOperations()) {
      StringRef name = ttirOperationName(&nested);
      if (name == "tt.load") {
        if (nested.getNumOperands() != 1 || nested.getNumResults() != 1) {
          nested.emitError("masked or multi-result loads are not supported");
          signalPassFailure();
          return;
        }
        auto arg = pointerArgument(nested.getOperand(0));
        if (!arg || *arg >= 3 || loads[*arg] || !isFp8Tile(nested.getResult(0))) {
          nested.emitError("MLP inputs must be unique row-major 32x32 FP8 loads");
          signalPassFailure();
          return;
        }
        loads[*arg] = &nested;
      } else if (name == "tt.dot") {
        dots.push_back(&nested);
      } else if (name == "tt.store") {
        if (store || nested.getNumOperands() != 2) {
          nested.emitError("expected one unmasked output store");
          signalPassFailure();
          return;
        }
        auto arg = pointerArgument(nested.getOperand(0));
        if (!arg || *arg != 3 || !isBf16Tile(nested.getOperand(1))) {
          nested.emitError("output must be one 32x32 BF16 store to Out");
          signalPassFailure();
          return;
        }
        store = &nested;
      }
    }
    if (!loads[0] || !loads[1] || !loads[2] || !store || dots.size() != 2) {
      ttFunc->emitError("expected three FP8 loads, two dots, and one BF16 store");
      signalPassFailure();
      return;
    }

    Operation *firstDot = dots[0];
    Operation *secondDot = dots[1];
    if (!isAcceptedDot(firstDot) || !isAcceptedDot(secondDot)) {
      ttFunc->emitError("dot types, zero accumulators, or IEEE precision are unsupported");
      signalPassFailure();
      return;
    }
    if (firstDot->getOperand(0) != loads[0]->getResult(0) ||
        firstDot->getOperand(1) != loads[1]->getResult(0) ||
        secondDot->getOperand(1) != loads[2]->getResult(0)) {
      ttFunc->emitError("dot dependencies do not match the two-layer MLP");
      signalPassFailure();
      return;
    }
    Operation *pack = secondDot->getOperand(0).getDefiningOp();
    if (!pack || ttirOperationName(pack) != "tt.fp_to_fp" ||
        !isFp8Tile(secondDot->getOperand(0)) ||
        // Triton 3.9.0 stores RTNE as I32EnumAttr value 1.
        !hasI32EnumAttribute(pack, "rounding", 1) ||
        !hasZeroReluPath(pack->getOperand(0), firstDot->getResult(0)) ||
        !dependsOn(store->getOperand(1), secondDot->getResult(0))) {
      ttFunc->emitError("expected BF16 ReLU, RTNE FP8 pack, and BF16 output truncation");
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    Location location = ttFunc->getLoc();
    OpBuilder builder(context);
    builder.setInsertionPoint(ttFunc);

    Type stateType = atlas::VirtualStateType::get(context);
    Type fp8Type = atlas::VirtualFP8Type::get(context);
    Type bf16Type = atlas::VirtualBF16Type::get(context);
    auto i32 = [&](int32_t value) { return builder.getI32IntegerAttr(value); };
    auto emit = [&](StringRef name, ValueRange operands, TypeRange results,
                    ArrayRef<NamedAttribute> attrs = {}) -> Operation * {
      OperationState state(location, name);
      state.addOperands(operands);
      state.addTypes(results);
      state.addAttributes(attrs);
      return builder.create(state);
    };

    Attribute symbolName = ttFunc->getAttr(SymbolTable::getSymbolAttrName());
    if (!symbolName) {
      ttFunc->emitError("tt.func has no symbol name");
      signalPassFailure();
      return;
    }
    OperationState functionState(location, "func.func");
    functionState.addAttribute(SymbolTable::getSymbolAttrName(), symbolName);
    functionState.addAttribute(
        "function_type",
        TypeAttr::get(builder.getFunctionType({}, TypeRange{stateType})));
    if (Attribute visibility = ttFunc->getAttr("sym_visibility"))
      functionState.addAttribute("sym_visibility", visibility);
    functionState.addAttribute("atlas.input_dram_base",
                               builder.getI64IntegerAttr(0x90000000));
    functionState.addAttribute("atlas.output_dram_base",
                               builder.getI64IntegerAttr(0x90002000));
    functionState.addAttribute("atlas.numerical_policy",
                               builder.getStringAttr("atlas-fp8-mlp-v1"));
    functionState.addRegion();
    Operation *atlasFunc = builder.create(functionState);
    Region &body = atlasFunc->getRegion(0);
    Block *entry = new Block();
    body.push_back(entry);
    builder.setInsertionPointToEnd(entry);

    Operation *start = emit("atlas.virtual_start", {}, TypeRange{stateType});
    Value io = start->getResult(0);
    std::array<Value, 3> inputTiles;
    for (int32_t index = 0; index < 3; ++index) {
      Operation *input = emit(
          "atlas.virtual_input_fp8", ValueRange{io},
          TypeRange{stateType, fp8Type},
          ArrayRef<NamedAttribute>{builder.getNamedAttr("index", i32(index))});
      io = input->getResult(0);
      inputTiles[index] = input->getResult(1);
    }
    Operation *weight0 = emit("atlas.virtual_xlu_transpose",
                              ValueRange{inputTiles[1]}, TypeRange{fp8Type});
    Operation *weight1 = emit("atlas.virtual_xlu_transpose",
                              ValueRange{inputTiles[2]}, TypeRange{fp8Type});
    Operation *hidden = emit(
        "atlas.virtual_mxu_matmul", ValueRange{inputTiles[0], weight0->getResult(0)},
        TypeRange{bf16Type},
        ArrayRef<NamedAttribute>{builder.getNamedAttr("unit", i32(0))});
    Operation *relu = emit(
        "atlas.virtual_vpu_unary", ValueRange{hidden->getResult(0)},
        TypeRange{bf16Type},
        ArrayRef<NamedAttribute>{builder.getNamedAttr("kind", builder.getStringAttr("relu"))});
    Operation *packed = emit(
        "atlas.virtual_pack_fp8", ValueRange{relu->getResult(0)},
        TypeRange{fp8Type},
        ArrayRef<NamedAttribute>{builder.getNamedAttr("scale_code", i32(127))});
    Operation *output = emit(
        "atlas.virtual_mxu_matmul", ValueRange{packed->getResult(0), weight1->getResult(0)},
        TypeRange{bf16Type},
        ArrayRef<NamedAttribute>{builder.getNamedAttr("unit", i32(1))});
    Operation *write = emit(
        "atlas.virtual_output_bf16", ValueRange{io, output->getResult(0)},
        TypeRange{stateType},
        ArrayRef<NamedAttribute>{builder.getNamedAttr("index", i32(0))});
    emit("func.return", ValueRange{write->getResult(0)}, TypeRange{});

    ttFunc->erase();
  }
};

} // namespace

} // namespace mlir::triton
