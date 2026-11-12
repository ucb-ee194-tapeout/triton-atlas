// Pass 3, convert-triton-compute-to-linalg. Contract: triton-atlas design
// agenda, page "Pass 3 · convert-triton-compute-to-linalg". Compute on float
// tensors becomes linalg on tensors, so MLIR's tiling can cut it into 32x32
// tiles in pass 4; loads, stores, pointers and address math stay TTIR. A
// partial conversion: only the ops the target marks illegal are rewritten.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
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

// tt.fp_to_fp -> arith.extf / arith.truncf, which the elementwise patterns
// then turn into a linalg.generic. Pass 1 lets through BF16 <-> f32 only, and
// truncf's default rounding is to nearest even, as tt.fp_to_fp's rtne.
struct FpToFpToArith : OpConversionPattern<FpToFpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto from = cast<FloatType>(getElementTypeOrSelf(op.getSrc().getType()));
    auto to = cast<FloatType>(getElementTypeOrSelf(op.getType()));
    // Equal widths (f16 <-> bf16) have no arith cast; a narrowing with a
    // rounding other than to-nearest-even isn't what arith.truncf does.
    if (from.getWidth() == to.getWidth())
      return rewriter.notifyMatchFailure(op, "no arith cast between equal "
                                             "widths");
    if (op.getRounding() && *op.getRounding() != RoundingMode::RTNE)
      return rewriter.notifyMatchFailure(op, "rounding is not rtne");
    Operation *cast;
    if (to.getWidth() > from.getWidth())
      cast = arith::ExtFOp::create(rewriter, op.getLoc(), op.getType(),
                                   adaptor.getSrc());
    else
      cast = arith::TruncFOp::create(rewriter, op.getLoc(), op.getType(),
                                     adaptor.getSrc());
    copyTags(op, cast);
    rewriter.replaceOp(op, cast->getResults());
    return success();
  }
};

// The dimensions a tt.broadcast stretches from size 1.
SmallVector<int64_t> broadcastDims(BroadcastOp op) {
  auto from = cast<RankedTensorType>(op.getSrc().getType());
  auto to = cast<RankedTensorType>(op.getType());
  SmallVector<int64_t> dims;
  for (int64_t i = 0, e = to.getRank(); i < e; ++i)
    if (from.getDimSize(i) != to.getDimSize(i))
      dims.push_back(i);
  return dims;
}

// Whether `broadcast` stretches exactly the dimension `expand` added, so the
// pair becomes one linalg.broadcast of the expand's source.
bool absorbs(BroadcastOp broadcast, ExpandDimsOp expand) {
  return broadcastDims(broadcast) ==
         ArrayRef<int64_t>{int64_t(expand.getAxis())};
}

// tt.broadcast of floats -> one linalg.broadcast. linalg.broadcast adds
// dimensions instead of stretching size-1 ones, so its input is the source
// without the dimensions being broadcast: the value under a tt.expand_dims
// that added exactly those (the bias case; tensor.expand_shape then
// linalg.broadcast doesn't verify, so the pair becomes one linalg.broadcast),
// or else the source with those unit dimensions collapsed away.
struct BroadcastToLinalg : OpConversionPattern<BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto from = cast<RankedTensorType>(op.getSrc().getType());
    auto to = cast<RankedTensorType>(op.getType());
    SmallVector<int64_t> dims = broadcastDims(op);
    if (dims.empty()) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    Value input;
    if (auto expand = op.getSrc().getDefiningOp<ExpandDimsOp>();
        expand && absorbs(op, expand)) {
      input = rewriter.getRemappedValue(expand.getSrc());
    } else {
      SmallVector<int64_t> shape;
      for (int64_t i = 0, e = from.getRank(); i < e; ++i)
        if (!llvm::is_contained(dims, i))
          shape.push_back(from.getDimSize(i));
      auto collapsed = RankedTensorType::get(shape, from.getElementType());
      std::optional<SmallVector<ReassociationIndices>> groups =
          getReassociationIndicesForCollapse(from.getShape(), shape);
      if (!groups)
        return rewriter.notifyMatchFailure(op, "can't drop the unit dims");
      input = tensor::CollapseShapeOp::create(rewriter, loc, collapsed,
                                              adaptor.getSrc(), *groups);
    }
    auto broadcast = linalg::BroadcastOp::create(
        rewriter, loc, input, emptyLike(rewriter, loc, to), dims);
    copyTags(op, broadcast);
    rewriter.replaceOp(op, broadcast->getResults());
    return success();
  }
};

// tt.expand_dims of floats that no broadcast absorbs -> tensor.expand_shape.
// Known limit: neither this nor the tensor.collapse_shape BroadcastToLinalg
// makes for a broadcast without a matching expand_dims has a lowering later
// (pass 4 can't tile a reshape, pass 7 doesn't match it), so such kernels fail
// in pass 7. The bias pattern (expand_dims then broadcast of the added
// dimension) never needs them.
struct ExpandDimsToExpandShape : OpConversionPattern<ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto from = cast<RankedTensorType>(op.getSrc().getType());
    auto to = cast<RankedTensorType>(op.getType());
    std::optional<SmallVector<ReassociationIndices>> groups =
        getReassociationIndicesForReshape(from, to);
    if (!groups)
      return rewriter.notifyMatchFailure(op, "no reassociation");
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, to, adaptor.getSrc(),
                                                       *groups);
    return success();
  }
};

// tt.splat of a float scalar -> linalg.fill. (A splat of a constant is
// usually an arith.constant already.)
struct SplatToFill : OpConversionPattern<SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto fill = linalg::FillOp::create(rewriter, loc, adaptor.getSrc(),
                                       emptyLike(rewriter, loc, op.getType()));
    copyTags(op, fill);
    rewriter.replaceOp(op, fill->getResults());
    return success();
  }
};

// Whether a tt.expand_dims only feeds broadcasts that absorb it; then it
// disappears with them. Any other float expand_dims becomes
// tensor.expand_shape.
bool absorbedByBroadcasts(ExpandDimsOp op) {
  return !op->use_empty() && llvm::all_of(op->getUsers(), [&](Operation *user) {
    auto broadcast = dyn_cast<BroadcastOp>(user);
    return broadcast && absorbs(broadcast, op);
  });
}

// Whether an arith / math op is elementwise compute on float tensors, which
// becomes a linalg.generic. Integer tensors are address math and stay TTIR:
// upstream's patterns would convert them too.
bool isFloatElementwise(Operation *op) {
  return OpTrait::hasElementwiseMappableTraits(op) &&
         llvm::any_of(llvm::concat<const Type>(op->getOperandTypes(),
                                               op->getResultTypes()),
                      isFloatTensor);
}

// Upstream's elementwise patterns reuse an operand as the generic's output
// when the body ignores it. Tile-and-fuse then duplicates that operand's
// producer (32 matmuls instead of 8 on a 64x64 MLP), so each such output
// becomes a fresh tensor.empty. A walk after the conversion, because
// upstream's patterns offer no hook to choose the output.
void freshOutputs(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  module.walk([&](linalg::GenericOp generic) {
    for (OpOperand &init : generic.getDpsInitsMutable()) {
      if (generic.payloadUsesValueFromOperand(&init) ||
          init.get().getDefiningOp<tensor::EmptyOp>())
        continue;
      rewriter.setInsertionPoint(generic);
      Value empty = emptyLike(rewriter, generic.getLoc(), init.get().getType());
      rewriter.modifyOpInPlace(generic, [&] { init.set(empty); });
    }
  });
}

// Upstream copies an arith op's attributes onto the scalar op inside the
// generic's body, so the ttatlas tags land there; they belong on the generic.
// A walk after the conversion, for the same reason as freshOutputs.
void hoistTags(ModuleOp module) {
  module.walk([](linalg::GenericOp generic) {
    for (Operation &op : generic.getBody()->without_terminator()) {
      copyTags(&op, generic);
      for (NamedAttribute attr : llvm::to_vector(op.getDiscardableAttrs()))
        if (attr.getName().strref().starts_with("ttatlas."))
          op.removeDiscardableAttr(attr.getName());
    }
  });
}

// Erases every trivially dead op in the module with MLIR's region DCE, which
// removes chains of dead ops too. The ones that matter are what the conversion
// left dead: the float expand_dims the broadcasts absorbed (the stage-3
// contract allows no float tt op) and the zero tensor constants a dot started
// from (the matmul gets a linalg.fill instead). Any other dead op goes as well,
// an unused tt.load included (it only reads), as canonicalize would remove it.
// Done here because no canonicalize runs between this pass and pass 4
// (Pipeline.cpp cleans up after passes 2, 4 and 5), so pass 4 sees only live
// IR. The kernel's arguments stay.
void eraseDeadLeftovers(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  (void)runRegionDCE(rewriter, module->getRegions());
}

struct ConvertTritonComputeToLinalg
    : impl::ConvertTritonComputeToLinalgBase<ConvertTritonComputeToLinalg> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect>();
    // Compute on float tensors only: a transpose of offsets or pointers is
    // address math and stays TTIR.
    target.addDynamicallyLegalOp<DotOp, TransOp>([](Operation *op) {
      return !isFloatTensor(op->getResult(0).getType());
    });
    target.addDynamicallyLegalOp<FpToFpOp, BroadcastOp, SplatOp>(
        [](Operation *op) {
          return !isFloatTensor(op->getResult(0).getType());
        });
    target.addDynamicallyLegalOp<ExpandDimsOp>([](ExpandDimsOp op) {
      return !isFloatTensor(op.getType()) || absorbedByBroadcasts(op);
    });
    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) { return !isFloatElementwise(op); });
    // Everything else stays as it is: loads, stores, pointers, address math.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<DotToMatmul, TransToTranspose, BroadcastToLinalg,
                 ExpandDimsToExpandShape, SplatToFill>(context);
    // Ahead of upstream's patterns, which would match tt.fp_to_fp too (it is
    // elementwise) and put it, unconverted, into a generic's body.
    patterns.add<FpToFpToArith>(context, /*benefit=*/2);
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
    eraseDeadLeftovers(getOperation());
    freshOutputs(getOperation());
    hoistTags(getOperation());
  }
};

} // namespace

} // namespace mlir::triton
