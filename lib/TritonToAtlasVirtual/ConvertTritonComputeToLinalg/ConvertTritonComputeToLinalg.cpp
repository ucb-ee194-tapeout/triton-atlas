// Pass 3, convert-triton-compute-to-linalg. Contract: triton-atlas design
// agenda, page "Pass 3 · convert-triton-compute-to-linalg". Compute on float
// tensors becomes linalg on tensors, so MLIR's tiling can cut it into 32x32
// tiles in pass 4; loads, stores, pointers and address math stay TTIR. A
// partial conversion: only the ops the target marks illegal are rewritten.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONCOMPUTETOLINALG
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;

// Copies the ttatlas tags of `from` onto `to`, except ttatlas.k_chain: from
// here on a K chain is the def-use chain fill -> matmul -> matmul itself, so
// the tag ends at this pass.
void copyTags(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getDiscardableAttrs())
    if (attr.getName().strref().starts_with("ttatlas.") &&
        attr.getName() != tta::kKChain)
      to->setAttr(attr.getName(), attr.getValue());
}

bool isFloatTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isa<FloatType>(tensor.getElementType());
}

// A fresh tensor.empty shaped like `type`.
Value emptyLike(OpBuilder &builder, Location loc, Type type) {
  auto tensor = cast<RankedTensorType>(type);
  return tensor::EmptyOp::create(builder, loc, tensor.getShape(),
                                 tensor.getElementType());
}

// tt.dot -> linalg.matmul. A dot that starts from zero gets linalg.fill zero
// as its output; a later step of a K chain accumulates onto the previous
// step's matmul, which links the chain pass 2 tagged into one
// fill -> matmul -> matmul def-use chain.
struct DotToMatmul : OpConversionPattern<DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (cast<RankedTensorType>(op.getType()).getRank() != 2)
      return rewriter.notifyMatchFailure(op, "not a 2-D dot");
    Location loc = op.getLoc();
    Value init = adaptor.getC();
    if (matchPattern(op.getC(), m_AnyZeroFloat())) {
      Type element = getElementTypeOrSelf(op.getType());
      Value zero = arith::ConstantOp::create(rewriter, loc, element,
                                             rewriter.getZeroAttr(element));
      init = linalg::FillOp::create(rewriter, loc, zero,
                                    emptyLike(rewriter, loc, op.getType()))
                 .getResult(0);
    }
    // tt.dot's inputPrecision and maxNumImpreciseAcc are dropped: they choose
    // how a GPU multiplies f32 (tf32, imprecise accumulation), and Atlas's MXU
    // multiplies FP8 E4M3 only (pass 1), so they have no meaning here. The
    // matmul_relu fixture carries inputPrecision = tf32, for example.
    auto matmul = linalg::MatmulOp::create(
        rewriter, loc, TypeRange{op.getType()},
        ValueRange{adaptor.getA(), adaptor.getB()}, ValueRange{init});
    copyTags(op, matmul);
    rewriter.replaceOp(op, matmul.getResult(0));
    return success();
  }
};

// tt.trans -> linalg.transpose, with the same permutation.
struct TransToTranspose : OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<int64_t> permutation(op.getOrder());
    auto transpose = linalg::TransposeOp::create(
        rewriter, loc, adaptor.getSrc(), emptyLike(rewriter, loc, op.getType()),
        permutation);
    copyTags(op, transpose);
    rewriter.replaceOp(op, transpose.getResult());
    return success();
  }
};

struct ConvertTritonComputeToLinalg
    : impl::ConvertTritonComputeToLinalgBase<ConvertTritonComputeToLinalg> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect>();
    // Compute on float tensors only: a transpose of offsets or pointers is
    // address math and stays TTIR.
    target.addDynamicallyLegalOp<DotOp, TransOp>([](Operation *op) {
      return !isFloatTensor(op->getResult(0).getType());
    });
    // Everything else stays as it is: loads, stores, pointers, address math.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<DotToMatmul, TransToTranspose>(context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
