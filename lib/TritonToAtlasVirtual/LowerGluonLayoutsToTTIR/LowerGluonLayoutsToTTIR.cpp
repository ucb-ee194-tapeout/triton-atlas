// G1: convert Gluon's layout-carrying tensor types to the layout-free TTIR
// contract used by Atlas.  G0 has already established that anchors are static
// and rank 1/2; this pass deliberately refuses regions until their block
// argument conversion is implemented.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
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

struct EraseAnchor : OpConversionPattern<gluon::SetAutoLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(gluon::SetAutoLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Gluon kernels commonly expose an encoded tensor at the function boundary.
// This conversion is one-to-one, so changing each entry argument in place is
// safe; region-bearing *inner* operations remain deliberately unsupported.
struct EraseFunctionSignature : OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(FuncOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> inputs;
    SmallVector<Type> results;
    if (failed(getTypeConverter()->convertTypes(op.getArgumentTypes(), inputs)) ||
        failed(getTypeConverter()->convertTypes(op.getResultTypes(), results)))
      return failure();
    rewriter.updateRootInPlace(op, [&] {
      for (auto [argument, type] : llvm::zip(op.getBody().front().getArguments(), inputs))
        argument.setType(type);
      op.setFunctionType(FunctionType::get(op.getContext(), inputs, results));
    });
    return success();
  }
};

// Rebuild region-free operations with converted operands and result types.
// This is intentionally generic: once a Gluon program has had layout removed,
// its ordinary tt/arith operations retain their original names and attributes.
struct EraseLayoutTypes : ConversionPattern {
  EraseLayoutTypes(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op->getRegions().empty())
      return failure();
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    SmallVector<Type> results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), results)))
      return failure();
    state.addTypes(results);
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

struct LowerGluonLayoutsToTTIR
    : impl::LowerGluonLayoutsToTTIRBase<LowerGluonLayoutsToTTIR> {
  void runOnOperation() override {
    TypeConverter converter;
    converter.addConversion([](Type type) { return eraseEncoding(type); });

    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
      return llvm::all_of(op.getArgumentTypes(), isLayoutFree) &&
             llvm::all_of(op.getResultTypes(), isLayoutFree);
    });
    target.addDynamicallyLegalOp<gluon::SetAutoLayoutOp>([](auto) { return false; });
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() != "gluon" &&
             isLayoutFreeOperation(op);
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<EraseAnchor>(converter, &getContext());
    patterns.add<EraseFunctionSignature>(converter, &getContext());
    patterns.add<EraseLayoutTypes>(converter, &getContext());
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton
