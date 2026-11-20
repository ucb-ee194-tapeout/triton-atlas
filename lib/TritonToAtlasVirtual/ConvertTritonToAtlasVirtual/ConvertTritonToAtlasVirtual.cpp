// Pass 7: stage-6 linalg + TTIR -> Atlas virtual IR and the kernel metadata.
// Upstream's specialization turns the generics pass 4 left back into named ops;
// then one ordered walk (Emitter.cpp) builds the argument-free Atlas function
// beside each tt.func, and KernelMetadata.cpp describes its boundary. Nothing
// is converted in place: the tt.func goes once its Atlas function exists.
#include "Emitter.h"
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "llvm/Support/JSON.h"

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONTOATLASVIRTUAL
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

// Private Python handoff, needed because Triton's IR bindings expose only i32
// and string attribute builders. Normalize to the module tags.
LogicalResult applyBackendOptions(ModuleOp module) {
  auto options = module->getAttrOfType<StringAttr>("atlas_backend.options");
  if (!options)
    return success();
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
          module, "backend placement options disagree with module attributes");
    module->setAttr(name, value);
  }
  module->removeAttr("atlas_backend.options");
  return success();
}

struct ConvertTritonToAtlasVirtual
    : impl::ConvertTritonToAtlasVirtualBase<ConvertTritonToAtlasVirtual> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(applyBackendOptions(module)))
      return signalPassFailure();
    SmallVector<FuncOp> functions(module.getOps<FuncOp>());
    for (FuncOp function : functions) {
      // One generic at a time: a greedy driver would also merge equal constants
      // and drop the slots on them.
      IRRewriter rewriter(&getContext());
      SmallVector<linalg::GenericOp> generics;
      function.walk(
          [&](linalg::GenericOp generic) { generics.push_back(generic); });
      for (linalg::GenericOp generic : generics) {
        rewriter.setInsertionPoint(generic);
        (void)linalg::specializeGenericOp(
            rewriter, generic); // a generic with no named form stays as it is
      }
      SlotTables tables;
      FailureOr<func::FuncOp> converted = emitAtlasFunction(function, tables);
      if (failed(converted) || failed(attachKernelMetadata(*converted, tables)))
        return signalPassFailure();
      function.erase();
    }
  }
};

} // namespace

} // namespace mlir::triton
