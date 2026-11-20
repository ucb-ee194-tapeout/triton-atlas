// Pass 7: the DRAM bases and ttatlas.kernel (schema v2), from the slots the
// emission walk read.
#include "Emitter.h"
#include "TTAtlas/AtlasTarget.h"

#include "Atlas/AtlasBoundaryABI.h"
#include "llvm/Support/JSON.h"

namespace mlir::triton {
namespace {

std::string typeName(Type type) {
  std::string text;
  llvm::raw_string_ostream(text) << type;
  return text;
}

llvm::json::Array array(ArrayRef<int64_t> values) {
  llvm::json::Array out;
  for (int64_t value : values)
    out.push_back(value);
  return out;
}

llvm::json::Object slotJson(ttatlas::SlotAttr slot) {
  llvm::json::Object out;
  bool constant = slot.getKind() == ttatlas::SlotKind::constant;
  out["index"] = slot.getIndex();
  out["kind"] = ttatlas::stringifySlotKind(slot.getKind());
  out["role"] = ttatlas::stringifySlotRole(slot.getRole());
  out["vector_layout"] = ttatlas::stringifyVectorLayout(slot.getVectorLayout());
  out["transposed"] = slot.getTransposed();
  out["host_dtype"] = typeName(slot.getHostDtype());
  out["tile_dtype"] = typeName(slot.getTileDtype());
  out["block"] = array(slot.getBlock());
  out["valid"] = array(slot.getValid());
  if (constant) {
    llvm::json::Array pattern;
    for (const APInt &bits : slot.getPattern().getValues<APInt>())
      pattern.push_back(static_cast<int64_t>(bits.getZExtValue()));
    out["pattern"] = std::move(pattern);
    return out;
  }
  out["arg"] = slot.getArg();
  out["offset"] = slot.getOffset();
  out["row_stride"] = slot.getRowStride();
  out["col_stride"] = slot.getColStride();
  // The raw bits of `other` in the tile dtype (it may be written in the host
  // dtype); absent means 0.
  if (FloatAttr other = slot.getOther()) {
    APFloat value = other.getValue();
    bool lost;
    value.convert(cast<FloatType>(slot.getTileDtype()).getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &lost);
    out["other"] = static_cast<int64_t>(value.bitcastToAPInt().getZExtValue());
  } else {
    out["other"] = nullptr;
  }
  return out;
}

} // namespace

LogicalResult attachKernelMetadata(func::FuncOp function,
                                   const SlotTables &tables) {
  auto fail = [&](const Twine &reason) {
    return ttatlas::emitUnsupported(function, reason);
  };
  for (const auto *table : {&tables.inputs, &tables.outputs}) {
    int32_t expected = 0;
    for (const auto &entry : *table)
      if (entry.first != expected++)
        return fail("slot indices must be dense from zero (a slot no tile "
                    "reads leaves a gap)");
  }
  if (tables.inputs.empty() || tables.outputs.empty())
    return fail("placement needs at least one input and one output slot");

  ModuleOp module = function->getParentOfType<ModuleOp>();
  int64_t base = ttatlas::target::kDefaultDramBase;
  if (Attribute attr = module->getAttr(ttatlas::kDramBase)) {
    auto integer = dyn_cast<IntegerAttr>(attr);
    if (!integer || !integer.getType().isSignlessInteger(64))
      return fail("ttatlas.dram_base must be i64");
    base = integer.getInt();
  }
  if (base < 0 || base > 0xffffffffLL ||
      tables.inputs.size() >
          (0x100000000ULL - base) / ttatlas::target::kSlotBytes)
    return fail("input slot window exceeds 32-bit Atlas DRAM");
  int64_t outputBase = base + static_cast<int64_t>(tables.inputs.size()) *
                                  ttatlas::target::kSlotBytes;
  Builder builder(function.getContext());
  function->setAttr("atlas.input_dram_base", builder.getI64IntegerAttr(base));
  function->setAttr("atlas.output_dram_base",
                    builder.getI64IntegerAttr(outputBase));
  auto abi = atlas::AtlasBoundaryABI::read(function);
  if (failed(abi) || failed(abi->verifySpans(function)))
    return failure();

  llvm::json::Object manifest;
  manifest["version"] = 2;
  manifest["name"] = function.getSymName();
  manifest["dram_base"] =
      llvm::json::Object{{"input", base}, {"output", outputBase}};
  for (auto [name, table] :
       {std::pair<StringRef, const std::map<int32_t, ttatlas::SlotAttr> *>{
            "inputs", &tables.inputs},
        {"outputs", &tables.outputs}}) {
    llvm::json::Array slots;
    for (const auto &entry : *table)
      slots.push_back(slotJson(entry.second));
    manifest[name] = std::move(slots);
  }
  manifest["grid"] = llvm::json::Array{1, 1, 1};
  std::string text;
  llvm::raw_string_ostream(text) << llvm::json::Value(std::move(manifest));
  function->setAttr(ttatlas::kKernel, builder.getStringAttr(text));
  return success();
}

} // namespace mlir::triton
