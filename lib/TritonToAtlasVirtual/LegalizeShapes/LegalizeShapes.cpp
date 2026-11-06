// First slice of pass 4: validate tensors already in Atlas tile shape and tag
// matmuls for the policy-selected MXU. Tiling and reassembly follow later.
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZESHAPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

static bool isLinalgCompute(Operation *op) {
  return op->getName().getDialectNamespace() == "linalg";
}

static bool isAtlasTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return false;
  if (tensor.getRank() == 1)
    return tensor.getDimSize(0) == ttatlas::target::kTileSize;
  return tensor.getRank() == 2 &&
         tensor.getDimSize(0) == ttatlas::target::kTileSize &&
         tensor.getDimSize(1) == ttatlas::target::kTileSize;
}

struct LegalizeShapes : impl::LegalizeShapesBase<LegalizeShapes> {
  void runOnOperation() override {
    bool failed = false;
    getOperation().walk([&](Operation *op) {
      if (!isLinalgCompute(op))
        return;

      for (Value value : op->getOperands()) {
        if (isa<TensorType>(value.getType()) && !isAtlasTile(value.getType())) {
          ttatlas::emitUnsupported(op, "expected linalg tensors to be 32 or 32x32");
          failed = true;
          return;
        }
      }
      for (Value value : op->getResults()) {
        if (isa<TensorType>(value.getType()) && !isAtlasTile(value.getType())) {
          ttatlas::emitUnsupported(op, "expected linalg tensors to be 32 or 32x32");
          failed = true;
          return;
        }
      }

      if (op->getName().getStringRef() != "linalg.matmul")
        return;

      Attribute oldUnit = op->getAttr(ttatlas::kMxu);
      auto oldInteger = dyn_cast_or_null<IntegerAttr>(oldUnit);
      if (oldUnit && (!oldInteger || oldInteger.getInt() != 0)) {
        ttatlas::emitUnsupported(op, "linalg.matmul has conflicting ttatlas.mxu; expected 0");
        failed = true;
        return;
      }
      // atlas-hw: MXU0 is the pass 4 default policy; MXU1 cannot run a K chain.
      op->setAttr(ttatlas::kMxu,
                  IntegerAttr::get(IntegerType::get(op->getContext(), 32), 0));
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
