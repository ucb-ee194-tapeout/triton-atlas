// Pass 7 patterns for a row linalg.reduce and its broadcast back to 32x32.
#include "Patterns.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {
namespace {

static StringRef reductionKind(Operation *op) {
  if (op->getNumRegions() != 1 || op->getRegion(0).empty())
    return {};
  for (Operation &nested : op->getRegion(0).front()) {
    StringRef name = nested.getName().getStringRef();
    if (name == "arith.addf" || name == "arith.addi")
      return "row_sum";
    if (name == "arith.maxnumf" || name == "arith.maximumf" ||
        name == "arith.maxsi" || name == "arith.maxui")
      return "row_max";
  }
  return {};
}

struct ConvertRowReduce : ConversionPattern {
  ConvertRowReduce(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.reduce", PatternBenefit(1),
                          context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || operands.empty() ||
        op->getNumResults() != 1)
      return ttatlas::emitUnsupported(
          op, "expected a one-result row reduction with input and init");
    if (!isa<atlas::VirtualBF16Type>(operands.front().getType()))
      return ttatlas::emitUnsupported(
          op, "row-reduction input did not convert to a virtual BF16 tile");

    auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto dimensions = op->getAttrOfType<DenseI64ArrayAttr>("dimensions");
    if (!input || !result || input.getShape() != ArrayRef<int64_t>{32, 32} ||
        result.getShape() != ArrayRef<int64_t>{32} || !dimensions ||
        dimensions.asArrayRef() != ArrayRef<int64_t>{1})
      return ttatlas::emitUnsupported(
          op, "only a 32x32 axis-1 row reduction to tensor<32xT> is supported");

    StringRef kind = reductionKind(op);
    if (kind.empty())
      return ttatlas::emitUnsupported(
          op, "row reduction requires an add or maximum body");
    OperationState state(op->getLoc(),
                         atlas::VirtualVPUReduceOp::getOperationName());
    state.addOperands(operands.front());
    state.addTypes(atlas::VirtualBF16Type::get(op->getContext()));
    state.addAttribute("kind", rewriter.getStringAttr(kind));
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

} // namespace

void populateReducePatterns(TypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  patterns.add<ConvertRowReduce>(typeConverter, patterns.getContext());
}

} // namespace mlir::triton
