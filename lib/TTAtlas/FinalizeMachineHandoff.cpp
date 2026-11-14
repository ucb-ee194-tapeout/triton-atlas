// TTAtlas owns its frontend annotations. Drop only obsolete module tags once
// Atlas has materialized and verified the machine program and its contracts.
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::ttatlas {
namespace {
struct FinalizeMachineHandoff
    : PassWrapper<FinalizeMachineHandoff, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeMachineHandoff)

  StringRef getArgument() const final {
    return "ttatlas-finalize-machine-handoff";
  }
  StringRef getDescription() const final {
    return "Remove obsolete TTAtlas module metadata from a verified timed "
           "Atlas machine program";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<atlas::AtlasDialect, TTAtlasDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(atlas::requireAtlasGeneratedArtifact(module)) ||
        failed(atlas::verifyAtlasTimingState(module, /*requireTimed=*/true)))
      return signalPassFailure();

    SmallVector<StringRef> obsolete = {kLayout, kDramBase, kFoldBiasIntoDot,
                                       kPrecisionPolicy, kLegalized};
    bool invalid = false;
    module.walk([&](Operation *op) {
      if (op != module.getOperation() &&
          op->getName().getDialectNamespace() != "atlas") {
        op->emitError("machine handoff requires only Atlas machine operations");
        invalid = true;
      }
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name.starts_with("ttatlas.") &&
            (op != module.getOperation() ||
             !llvm::is_contained(obsolete, name))) {
          op->emitError("unexpected TTAtlas metadata at machine handoff: ")
              << name;
          invalid = true;
        }
      }
    });
    if (invalid)
      return signalPassFailure();
    SmallVector<uint32_t> words;
    if (failed(atlas::verifyAtlasArtifact(module, /*llvmBlock=*/true, words)))
      return signalPassFailure();

    // No operation, Atlas contract, address, or timing attribute is modified.
    // The driver retains the exported ttatlas.kernel manifest separately.
    for (StringRef name : obsolete)
      module->removeAttr(name);
  }
};
} // namespace

void registerFinalizeMachineHandoffPass() {
  static PassRegistration<FinalizeMachineHandoff> registration;
}
} // namespace mlir::ttatlas
