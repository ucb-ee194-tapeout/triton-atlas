// Pass 7 kernel ABI and metadata.
// The documented v1 field set is known, but tile_origin and host_dtype JSON
// encodings await the team's schema/golden example. Do not publish a partial or
// independently invented ttatlas.kernel. Derive/check the unambiguous ABI now.
#include "Patterns.h"
#include "TritonToAtlasVirtual/Passes.h"

#include "Atlas/AtlasBoundaryABI.h"
#include "Atlas/AtlasOps.h"
#include "llvm/Support/JSON.h"

#include <map>

namespace mlir::triton {
LogicalResult attachKernelMetadata(ModuleOp module) {
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!atlas::hasAtlasBoundaryTiles(function))
      continue;
    std::map<int32_t, ttatlas::SlotAttr> inputs, outputs;
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

    int64_t base = 0x90000000;
    if (auto attr = module->getAttrOfType<IntegerAttr>(ttatlas::kDramBase))
      base = attr.getInt();
    if (base < 0 || base > 0xffffffffLL ||
        inputs.size() > (0x100000000ULL - base) / 2048)
      return ttatlas::emitUnsupported(
          function, "input slot window exceeds 32-bit Atlas DRAM");
    int64_t outputBase = base + static_cast<int64_t>(inputs.size()) * 2048;
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

    // A supplied finalized manifest can be preserved/exported, but must agree
    // with fields whose serialization is already unambiguous. Absent a
    // manifest, stage=6 correctly remains incomplete until the schema decision
    // is made.
    if (auto manifest = function->getAttrOfType<StringAttr>(ttatlas::kKernel)) {
      auto parsed = llvm::json::parse(manifest.getValue());
      if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return ttatlas::emitUnsupported(function,
                                        "invalid ttatlas.kernel JSON");
      }
      const auto *object = parsed->getAsObject();
      const auto *bases = object ? object->getObject("dram_base") : nullptr;
      const auto *grid = object ? object->getArray("grid") : nullptr;
      const auto *in = object ? object->getArray("inputs") : nullptr;
      const auto *out = object ? object->getArray("outputs") : nullptr;
      if (!object || object->getInteger("version") != 1 ||
          object->getString("name") != function.getSymName() || !bases ||
          bases->getInteger("input") != base ||
          bases->getInteger("output") != outputBase || !grid ||
          grid->size() != 3 || (*grid)[0].getAsInteger() != 1 ||
          (*grid)[1].getAsInteger() != 1 || (*grid)[2].getAsInteger() != 1 ||
          !in || in->size() != inputs.size() || !out ||
          out->size() != outputs.size())
        return ttatlas::emitUnsupported(
            function, "ttatlas.kernel disagrees with function name, DRAM "
                      "bases, slot counts, or M1 grid");
      for (auto [array, table] :
           {std::make_pair(in, &inputs), std::make_pair(out, &outputs)}) {
        for (const auto &[index, slot] : *table) {
          const auto *entry = (*array)[index].getAsObject();
          if (!entry || entry->getInteger("index") != index ||
              entry->getInteger("arg") != slot.getArg() ||
              entry->getString("role") !=
                  ttatlas::stringifySlotRole(slot.getRole()))
            return ttatlas::emitUnsupported(
                function, "ttatlas.kernel slot index, argument, or role "
                          "disagrees with #ttatlas.slot");
        }
      }
    }
  }
  return success();
}
} // namespace mlir::triton
