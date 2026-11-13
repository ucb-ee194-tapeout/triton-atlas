// Pass 7 patterns for a row linalg.reduce and its broadcast back to 32x32.
#include "Patterns.h"

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "TTAtlas/TTAtlas.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

#include <optional>
#include <string>

namespace mlir::triton {
namespace {

// The row operation of a reduction whose body is one floating-point combiner,
// and the value that combiner starts from (its identity). Integer combiners
// have no VPU row operation: the VPU reduces BF16 and f32 only.
struct RowReduction {
  StringRef kind;
  llvm::APFloat identity;
};

static std::optional<RowReduction> rowReduction(Operation *op,
                                                std::string &why) {
  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      op->getRegion(0).front().getOperations().size() != 2) {
    why = "the reduction body must be one combiner and a yield";
    return std::nullopt;
  }
  Operation &combiner = op->getRegion(0).front().front();
  auto element = dyn_cast<FloatType>(combiner.getResult(0).getType());
  if (!element) {
    why = "an integer reduction has no VPU row operation (BF16 and f32 only)";
    return std::nullopt;
  }
  const llvm::fltSemantics &semantics = element.getFloatSemantics();
  if (isa<arith::AddFOp>(combiner))
    return RowReduction{"row_sum", llvm::APFloat::getZero(semantics)};
  if (isa<arith::MaxNumFOp, arith::MaximumFOp>(combiner))
    return RowReduction{"row_max",
                        llvm::APFloat::getInf(semantics, /*Negative=*/true)};
  if (isa<arith::MinNumFOp, arith::MinimumFOp>(combiner))
    return RowReduction{"row_min",
                        llvm::APFloat::getInf(semantics, /*Negative=*/false)};
  why = "the combiner is not an add, a maximum or a minimum";
  return std::nullopt;
}

// Is `init` a linalg.fill of exactly `identity`? virtual_vpu_reduce computes
// the reduction of the row alone, so a start value that is not the identity
// would be dropped.
static bool isIdentityFill(Value init, const llvm::APFloat &identity) {
  auto fill = init.getDefiningOp<linalg::FillOp>();
  if (!fill)
    return false;
  FloatAttr attribute;
  if (!matchPattern(fill.getInputs().front(), m_Constant(&attribute)))
    return false;
  const llvm::APFloat &value = attribute.getValue();
  return identity.isZero() ? value.isZero()
                           : (value.isInfinity() &&
                              value.isNegative() == identity.isNegative());
}

struct ConvertRowReduce : ConversionPattern {
  ConvertRowReduce(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "linalg.reduce", PatternBenefit(1),
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || operands.empty() ||
        op->getNumResults() != 1)
      return ttatlas::emitUnsupported(
          op, "expected a one-result row reduction with input and init");
    if (!isa<atlas::VirtualBF16Type>(operands.front().getType()))
      return ttatlas::emitUnsupported(
          op, "row-reduction input did not convert to a virtual BF16 tile");

    // The result may already carry the converted virtual type while this
    // pattern runs, and the input is already one virtual 32x32 tile. The
    // linalg init operand retains the logical row shape.
    auto result = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto dimensions = op->getAttrOfType<DenseI64ArrayAttr>("dimensions");
    if (!result || result.getShape() != ArrayRef<int64_t>{32} || !dimensions ||
        dimensions.asArrayRef() != ArrayRef<int64_t>{1})
      return ttatlas::emitUnsupported(
          op, "only a 32x32 axis-1 row reduction to tensor<32xT> is supported");

    std::string why;
    std::optional<RowReduction> reduction = rowReduction(op, why);
    if (!reduction)
      return ttatlas::emitUnsupported(op, why);
    if (!isIdentityFill(op->getOperand(1), reduction->identity))
      return ttatlas::emitUnsupported(
          op, "the reduction must start from a linalg.fill of its combiner's "
              "identity (0 for a sum, -inf for a max, +inf for a min): the "
              "virtual row reduction takes no start value");
    OperationState state(op->getLoc(),
                         atlas::VirtualVPUReduceOp::getOperationName());
    state.addOperands(operands.front());
    state.addTypes(atlas::VirtualBF16Type::get(op->getContext()));
    state.addAttribute("kind", rewriter.getStringAttr(reduction->kind));
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
