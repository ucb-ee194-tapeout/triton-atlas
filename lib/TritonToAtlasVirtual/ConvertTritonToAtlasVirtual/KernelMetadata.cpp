// Pass 7: deterministic ttatlas.kernel v1 and the Atlas boundary ABI.
#include "Patterns.h"
#include "TTAtlas/AtlasTarget.h"
#include "TritonToAtlasVirtual/Passes.h"

#include "Atlas/AtlasBoundaryABI.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/JSON.h"

#include <map>

namespace mlir::triton {
LogicalResult attachKernelMetadata(ModuleOp module) {
  // Private Python handoff, needed because Triton's IR bindings expose only
  // i32 and string attribute builders. Normalize to the frozen module tags.
  if (auto options =
          module->getAttrOfType<StringAttr>("atlas_backend.options")) {
    auto parsed = llvm::json::parse(options.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return ttatlas::emitUnsupported(module,
                                      "invalid backend placement options");
    }
    auto *object = parsed->getAsObject();
    auto base = object ? object->getInteger("dram_base") : std::nullopt;
    auto layoutName = object ? object->getString("layout") : std::nullopt;
    auto layout =
        layoutName ? ttatlas::symbolizeLayout(*layoutName) : std::nullopt;
    if (!base || !layout)
      return ttatlas::emitUnsupported(module,
                                      "invalid backend placement options");
    Builder builder(module.getContext());
    for (auto [name, value] :
         {std::pair<StringRef, Attribute>{ttatlas::kDramBase,
                                          builder.getI64IntegerAttr(*base)},
          {ttatlas::kLayout,
           ttatlas::LayoutAttr::get(module.getContext(), *layout)}}) {
      if (Attribute existing = module->getAttr(name);
          existing && existing != value)
        return ttatlas::emitUnsupported(
            module,
            "backend placement options disagree with module attributes");
      module->setAttr(name, value);
    }
    module->removeAttr("atlas_backend.options");
  }
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!atlas::hasAtlasBoundaryTiles(function))
      continue;
    std::map<int32_t, ttatlas::SlotAttr> inputs, outputs;
    std::map<int32_t, ttatlas::PadAttr> padding;
    bool invalid = false;
    function.walk([&](Operation *op) {
      bool input = isa<atlas::VirtualInputBF16Op, atlas::VirtualInputFP8Op>(op);
      if (!input && !isa<atlas::VirtualOutputBF16Op>(op))
        return;
      auto slot = op->getAttrOfType<ttatlas::SlotAttr>(ttatlas::kSlot);
      auto index = op->getAttrOfType<IntegerAttr>("index");
      if (!slot || !index || slot.getIndex() != index.getInt()) {
        ttatlas::emitUnsupported(op, "virtual boundary index disagrees with "
                                     "its preserved #ttatlas.slot");
        invalid = true;
        return;
      }
      auto origin = slot.getTileOrigin().getValue();
      if (!llvm::all_of(origin.getResults(), [](AffineExpr expr) {
            auto constant = dyn_cast<AffineConstantExpr>(expr);
            return constant && constant.getValue() >= 0;
          })) {
        ttatlas::emitUnsupported(
            op, "M1 metadata requires constant nonnegative tile origins");
        invalid = true;
        return;
      }
      if (input)
        if (auto pad = op->getAttrOfType<ttatlas::PadAttr>(ttatlas::kPad))
          padding[slot.getIndex()] = pad;
      auto &table = input ? inputs : outputs;
      auto [it, inserted] = table.emplace(slot.getIndex(), slot);
      if (!inserted && (!input || it->second != slot)) {
        ttatlas::emitUnsupported(op, "inconsistent or duplicate boundary slot");
        invalid = true;
      }
    });
    if (invalid)
      return failure();
    for (const auto *table : {&inputs, &outputs}) {
      int32_t expected = 0;
      for (const auto &[index, slot] : *table)
        if (index != expected++)
          return ttatlas::emitUnsupported(
              function, "boundary slot indices must be dense from zero");
    }
    if (inputs.empty() || outputs.empty())
      return ttatlas::emitUnsupported(
          function, "M1 placement requires nonempty input/output tables; "
                    "zero-input window placement needs coordination");

    int64_t base = ttatlas::target::kDefaultDramBase;
    if (Attribute attr = module->getAttr(ttatlas::kDramBase)) {
      auto integer = dyn_cast<IntegerAttr>(attr);
      if (!integer || !integer.getType().isSignlessInteger(64))
        return ttatlas::emitUnsupported(function,
                                        "ttatlas.dram_base must be i64");
      base = integer.getInt();
    }
    if (base < 0 || base > 0xffffffffLL ||
        inputs.size() > (0x100000000ULL - base) / ttatlas::target::kSlotBytes)
      return ttatlas::emitUnsupported(
          function, "input slot window exceeds 32-bit Atlas DRAM");
    int64_t outputBase = base + static_cast<int64_t>(inputs.size()) *
                                    ttatlas::target::kSlotBytes;
    Builder builder(module.getContext());
    for (auto [name, expected] :
         {std::pair<StringRef, int64_t>{"atlas.input_dram_base", base},
          {"atlas.output_dram_base", outputBase}}) {
      if (Attribute existing = function->getAttr(name)) {
        auto integer = dyn_cast<IntegerAttr>(existing);
        if (!integer || !integer.getType().isSignlessInteger(64) ||
            integer.getInt() != expected)
          return ttatlas::emitUnsupported(
              function, "existing Atlas DRAM base disagrees with module "
                        "options and slot tables");
      }
      function->setAttr(name, builder.getI64IntegerAttr(expected));
    }
    auto abi = atlas::AtlasBoundaryABI::read(function);
    if (failed(abi) || failed(abi->verifySpans(function)))
      return failure();

    ttatlas::Layout layout = ttatlas::Layout::row_major;
    if (Attribute attr = module->getAttr(ttatlas::kLayout)) {
      auto typed = dyn_cast<ttatlas::LayoutAttr>(attr);
      if (!typed)
        return ttatlas::emitUnsupported(
            function, "ttatlas.layout must be a typed layout attribute");
      layout = typed.getValue();
    }
    ArrayAttr history;
    if (Attribute attr = module->getAttr(ttatlas::kLegalized)) {
      history = dyn_cast<ArrayAttr>(attr);
      if (!history)
        return ttatlas::emitUnsupported(function,
                                        "ttatlas.legalized must be an array");
      for (Attribute item : history) {
        auto entry = dyn_cast<DictionaryAttr>(item);
        if (!entry || entry.size() != 3 || !entry.getAs<StringAttr>("pass") ||
            !entry.getAs<StringAttr>("what") || !entry.getAs<StringAttr>("loc"))
          return ttatlas::emitUnsupported(
              function,
              "legalization entries require string pass, what and loc");
      }
    }
    std::string json;
    llvm::raw_string_ostream stream(json);
    llvm::json::OStream out(stream);
    auto table = [&](StringRef name, const auto &slots) {
      out.attributeArray(name, [&] {
        for (const auto &[index, slot] : slots) {
          std::string origin, dtype;
          llvm::raw_string_ostream(origin) << slot.getTileOrigin();
          llvm::raw_string_ostream(dtype) << slot.getHostDtype();
          out.object([&] {
            out.attribute("index", index);
            out.attribute("arg", slot.getArg());
            out.attribute("role", ttatlas::stringifySlotRole(slot.getRole()));
            out.attribute("tile_origin", origin);
            out.attributeArray("logical_shape", [&] {
              for (int64_t dim : slot.getLogicalShape())
                out.value(dim);
            });
            if (name == "inputs")
              if (auto pad = padding[index])
                out.attribute("pad_neutral",
                              ttatlas::stringifyPadNeutral(pad.getNeutral()));
            out.attribute("host_dtype", dtype);
            out.attribute("layout", ttatlas::stringifyLayout(layout));
          });
        }
      });
    };
    out.object([&] {
      out.attribute("version", 1);
      out.attribute("name", function.getSymName());
      out.attributeObject("dram_base", [&] {
        out.attribute("input", base);
        out.attribute("output", outputBase);
      });
      table("inputs", inputs);
      table("outputs", outputs);
      out.attributeArray("grid", [&] {
        out.value(1);
        out.value(1);
        out.value(1);
      });
      out.attributeArray("legalized", [&] {
        if (!history)
          return;
        for (Attribute item : history) {
          auto entry = cast<DictionaryAttr>(item);
          out.object([&] {
            for (StringRef key : {"pass", "what", "loc"})
              out.attribute(key, entry.getAs<StringAttr>(key).getValue());
          });
        }
      });
    });
    // A supplied manifest must match every field, including origins, original
    // host dtypes, layout and legalization history. Never export stale JSON.
    if (Attribute existing = function->getAttr(ttatlas::kKernel)) {
      auto manifest = dyn_cast<StringAttr>(existing);
      auto parsed = llvm::json::parse(manifest ? manifest.getValue() : "");
      auto generated = llvm::json::parse(json);
      if (!generated) {
        llvm::consumeError(generated.takeError());
        return ttatlas::emitUnsupported(function,
                                        "failed to serialize kernel metadata");
      }
      if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return ttatlas::emitUnsupported(function,
                                        "invalid ttatlas.kernel JSON");
      }
      if (*parsed != *generated)
        return ttatlas::emitUnsupported(
            function,
            "ttatlas.kernel disagrees with function name, DRAM bases, slot "
            "counts, or M1 grid; "
            "all slot fields and legalization history must also match");
    }
    function->setAttr(ttatlas::kKernel, builder.getStringAttr(json));
  }
  return success();
}
} // namespace mlir::triton
