// First slice of Atlas shape legalization. This pass validates tile shapes
// produced by pass 3 and records the MXU choice consumed by later lowering.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZEATLASSHAPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

static bool isLinalgCompute(Operation *op) {
  return op->getName().getDialectNamespace() == "linalg";
}

static bool isAtlasTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getRank() == 2 && tensor.getDimSize(0) == 32 &&
         tensor.getDimSize(1) == 32;
}

struct LegalizeAtlasShapes
    : impl::LegalizeAtlasShapesBase<LegalizeAtlasShapes> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool failed = false;

    module.walk([&](Operation *op) {
      if (!isLinalgCompute(op))
        return;

      for (Value value : op->getOperands()) {
        if (isa<TensorType>(value.getType()) && !isAtlasTile(value.getType())) {
          op->emitError("expected linalg compute tensor to be 32x32");
          failed = true;
          return;
        }
      }
      for (Value value : op->getResults()) {
        if (isa<TensorType>(value.getType()) && !isAtlasTile(value.getType())) {
          op->emitError("expected linalg compute tensor to be 32x32");
          failed = true;
          return;
        }
      }

      if (op->getName().getStringRef() != "linalg.matmul")
        return;

      Attribute oldUnit = op->getAttr("ttatlas.mxu");
      auto oldInteger = dyn_cast_or_null<IntegerAttr>(oldUnit);
      if (oldUnit && (!oldInteger || oldInteger.getInt() != 0)) {
        op->emitError("linalg.matmul has conflicting ttatlas.mxu; expected 0");
        failed = true;
        return;
      }
      op->setAttr("ttatlas.mxu",
                  IntegerAttr::get(IntegerType::get(op->getContext(), 32), 0));
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton
