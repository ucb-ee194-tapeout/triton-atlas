// G1: normalize GLIR's representation-only tensor layouts into the ordinary
// TTIR type contract consumed by Atlas. A Gluon layout changes distribution,
// not logical shape, element type, or SSA value flow.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_LOWERGLUONLAYOUTSTOTTIR
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

Type eraseEncoding(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type); tensor && tensor.getEncoding())
    return tensor.cloneWithEncoding({});
  return type;
}

bool isLayoutFree(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return !tensor || !tensor.getEncoding();
}

bool isLayoutFreeOperation(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), isLayoutFree) &&
         llvm::all_of(op->getResultTypes(), isLayoutFree);
}

// Layout removal is strictly one-to-one. Mutating result and block-argument
// types preserves regions, terminators, attributes, and SSA value flow; the
// previous generic dialect conversion rejected every region-bearing op.
void eraseAllTensorEncodings(ModuleOp module) {
  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(eraseEncoding(argument.getType()));
    for (Value result : op->getResults())
      result.setType(eraseEncoding(result.getType()));
  });

  module.walk([&](FuncOp func) {
    SmallVector<Type> inputs(func.getArgumentTypes());
    SmallVector<Type> results(func.getResultTypes());
    for (Type &input : inputs)
      input = eraseEncoding(input);
    for (Type &result : results)
      result = eraseEncoding(result);
    func.setFunctionType(FunctionType::get(func.getContext(), inputs, results));
  });
}

struct LowerGluonLayoutsToTTIR
    : impl::LowerGluonLayoutsToTTIRBase<LowerGluonLayoutsToTTIR> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // G0 proved anchors are logically type-preserving. Erase them before
    // normalizing all remaining SSA boundaries.
    SmallVector<gluon::SetAutoLayoutOp> anchors;
    module.walk([&](gluon::SetAutoLayoutOp op) { anchors.push_back(op); });
    for (gluon::SetAutoLayoutOp anchor : anchors)
      anchor.replaceAllUsesWith(anchor.getSrc());
    for (gluon::SetAutoLayoutOp anchor : anchors)
      anchor.erase();

    eraseAllTensorEncodings(module);

    bool failed = false;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "gluon" ||
          !isLayoutFreeOperation(op)) {
        op->emitError("Atlas Gluon layout normalization left an encoded value or Gluon operation");
        failed = true;
      }
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton
