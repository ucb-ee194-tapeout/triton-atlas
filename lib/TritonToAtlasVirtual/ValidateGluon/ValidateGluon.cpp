// G0: reject Gluon programs whose layout machinery cannot be erased into the
// layout-free TTIR contract consumed by the existing Atlas passes.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_VALIDATEGLUON
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

LogicalResult reject(Operation *op, const llvm::Twine &reason) {
  ttatlas::emitUnsupported(op, "Gluon ") << reason;
  return failure();
}

LogicalResult checkAnchor(gluon::SetAutoLayoutOp op) {
  auto source = dyn_cast<RankedTensorType>(op.getSrc().getType());
  auto result = dyn_cast<RankedTensorType>(op.getType());
  if (!source || !result)
    return reject(op, "set_auto_layout must convert ranked tensor types");
  if (source.getRank() < 1 || source.getRank() > 2 ||
      result.getRank() != source.getRank())
    return reject(op, "set_auto_layout has a tensor rank other than 1 or 2");
  if (!source.hasStaticShape() || !result.hasStaticShape())
    return reject(op, "set_auto_layout has a dynamic tensor shape");
  if (source.getShape() != result.getShape() ||
      source.getElementType() != result.getElementType())
    return reject(op, "set_auto_layout changes a tensor's logical shape or element type");
  if (!isa<gluon::AutoEncodingAttr>(source.getEncoding()))
    return reject(op, "set_auto_layout input is not an auto-layout tensor");
  if (!result.getEncoding())
    return reject(op, "set_auto_layout result has no layout encoding");
  return success();
}

struct ValidateGluon : impl::ValidateGluonBase<ValidateGluon> {
  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](Operation *op) {
      StringRef dialect = op->getName().getDialectNamespace();
      if (dialect == "ttg") {
        reject(op, "contains a TritonGPU operation before Atlas layout normalization");
        return WalkResult::interrupt();
      }
      if (dialect != "gluon")
        return WalkResult::advance();
      if (auto anchor = dyn_cast<gluon::SetAutoLayoutOp>(op))
        return succeeded(checkAnchor(anchor)) ? WalkResult::advance()
                                                : WalkResult::interrupt();
      reject(op, "operation has no Atlas layout-normalization rule");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton
