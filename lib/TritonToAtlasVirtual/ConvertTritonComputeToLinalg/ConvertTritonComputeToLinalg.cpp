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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

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

// Build a tensor linalg.fill and make its structured-op body explicit.  Recent
// MLIR FillOp builders normally materialize this body through
// buildStructuredOp, but the MLIR library loaded with Triton's Python runtime
// may leave the declared region without a block.  An empty region is invalid
// (and fails verification before later Atlas passes can run).  Keeping this
// here makes the pass independent of that builder/runtime mismatch while
// preserving the standard linalg.fill semantics.
linalg::FillOp makeFill(OpBuilder &builder, Location loc, Value value,
                        Value output) {
  // FillOp::create inserts (and verifies) the operation before returning it.
  // Construct the state first so the body exists at that verification point.
  OperationState state(loc, linalg::FillOp::getOperationName());
  linalg::FillOp::build(builder, state, ValueRange{value}, ValueRange{output});
  assert(state.regions.size() == 1 && "linalg.fill must have one region");
  auto fillOperation = Operation::create(state);
  auto fill = cast<linalg::FillOp>(fillOperation);
  Region &region = fill->getRegion(0);
  Block *body = region.empty() ? &region.emplaceBlock() : &region.front();
  body->addArgument(value.getType(), loc);
  body->addArgument(getElementTypeOrSelf(output.getType()), loc);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(body);
    builder.create<linalg::YieldOp>(loc, body->getArgument(0));
  }
  return cast<linalg::FillOp>(builder.insert(fillOperation));
}

// See makeFill: construct the linalg.matmul region before inserting the
// operation so verification never observes the empty region produced by the
// incompatible runtime builder.
linalg::MatmulOp makeMatmul(OpBuilder &builder, Location loc, Type result,
                            Value lhs, Value rhs, Value output) {
  OperationState state(loc, linalg::MatmulOp::getOperationName());
  linalg::MatmulOp::build(builder, state, TypeRange{result},
                          ValueRange{lhs, rhs}, ValueRange{output});
  assert(state.regions.size() == 1 && "linalg.matmul must have one region");
  auto matmulOperation = Operation::create(state);
  auto matmul = cast<linalg::MatmulOp>(matmulOperation);
  Region &region = matmul->getRegion(0);
  Block *body = region.empty() ? &region.emplaceBlock() : &region.front();
  SmallVector<Value> operands{lhs, rhs, output};
  for (Value operand : operands)
    body->addArgument(getElementTypeOrSelf(operand.getType()), loc);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(body);
    ImplicitLocOpBuilder regionBuilder(loc, builder);
    linalg::MatmulOp::regionBuilder(regionBuilder, *body, {}, {});
  }
  return cast<linalg::MatmulOp>(builder.insert(matmulOperation));
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
      init = makeFill(rewriter, loc, zero,
                      emptyLike(rewriter, loc, op.getType()))
                 .result();
    }
    // tt.dot's inputPrecision and maxNumImpreciseAcc are dropped: they choose
    // how a GPU multiplies f32 (tf32, imprecise accumulation), and Atlas's MXU
    // multiplies FP8 E4M3 only (pass 1), so they have no meaning here. The
    // matmul_relu test input carries inputPrecision = tf32, for example.
    auto matmul = makeMatmul(rewriter, loc, op.getType(), adaptor.getA(),
                             adaptor.getB(), init);
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
    // rounding other than to-nearest-even isn't what arith.truncf does. A
    // widening is exact, so its rounding mode doesn't matter (pass 1 and
    // checkCompute check rounding only on narrowing, too).
    if (from.getWidth() == to.getWidth())
      return rewriter.notifyMatchFailure(op, "no arith cast between equal "
                                             "widths");
    if (to.getWidth() < from.getWidth() && op.getRounding() &&
        *op.getRounding() != RoundingMode::RTNE)
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
    auto fill = makeFill(rewriter, loc, adaptor.getSrc(),
                         emptyLike(rewriter, loc, op.getType()));
    copyTags(op, fill);
    rewriter.replaceOp(op, fill->getResults());
    return success();
  }
};

// The value a row reduction starts from: the identity of its combine (-0.0
// for an add, the exact one, since +0.0 + -0.0 = +0.0; -inf for a max, +inf
// for a min). tt.reduce has no init value, and linalg.reduce needs one. Null
// for another combine (pass 1 rejects those).
TypedAttr reduceIdentity(Operation *combine, FloatType type) {
  if (isa<arith::AddFOp>(combine))
    return FloatAttr::get(type, -0.0);
  bool max = isa<arith::MaxNumFOp, arith::MaximumFOp>(combine);
  if (!max && !isa<arith::MinNumFOp, arith::MinimumFOp>(combine))
    return nullptr;
  return FloatAttr::get(
      type, APFloat::getInf(type.getFloatSemantics(), /*Negative=*/max));
}

// tt.reduce of one float tensor -> linalg.reduce over the same axis, its
// output filled with the combine's identity, its body the same combine.
struct ReduceToLinalg : OpConversionPattern<ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getCombineOp().front();
    auto type = dyn_cast<RankedTensorType>(op.getResult()[0].getType());
    if (op.getNumOperands() != 1 || !type || body.getOperations().size() != 2)
      return rewriter.notifyMatchFailure(op, "not one combine of one tensor");
    Operation *combine = &body.front();
    auto args = body.getArguments();
    bool onArgs = combine->getNumOperands() == 2 &&
                  ((combine->getOperand(0) == args[0] &&
                    combine->getOperand(1) == args[1]) ||
                   (combine->getOperand(0) == args[1] &&
                    combine->getOperand(1) == args[0]));
    if (!onArgs || body.getTerminator()->getOperand(0) != combine->getResult(0))
      return rewriter.notifyMatchFailure(op, "the body isn't one combine of "
                                             "its two arguments");
    TypedAttr identity =
        reduceIdentity(combine, cast<FloatType>(type.getElementType()));
    if (!identity)
      return rewriter.notifyMatchFailure(op, "no identity for this combine");
    Location loc = op.getLoc();
    Value start = arith::ConstantOp::create(rewriter, loc, identity);
    Value init = makeFill(rewriter, loc, start,
                          emptyLike(rewriter, loc, type))
                     .result();
    auto reduce = linalg::ReduceOp::create(
        rewriter, loc, adaptor.getSrcs(), ValueRange{init},
        SmallVector<int64_t>{op.getAxis()},
        [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {
          // linalg.reduce's body takes (element, accumulator); Triton's
          // combine takes the accumulator first.
          IRMapping mapping;
          mapping.map(body.getArgument(0), blockArgs[1]);
          mapping.map(body.getArgument(1), blockArgs[0]);
          Operation *step = builder.clone(*combine, mapping);
          linalg::YieldOp::create(builder, bodyLoc, step->getResult(0));
        });
    copyTags(op, reduce);
    rewriter.replaceOp(op, reduce->getResults());
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

bool isScalarElementwiseType(Type type) {
  return isa<IntegerType, FloatType, IndexType, ComplexType>(type);
}

// The upstream elementwise-to-linalg pattern calls GenericOp::create, which
// inserts the op before its body builder runs.  With the MLIR implementation
// supplied by Triton's runtime, that fails verification on an empty region.
// Build the exact same identity-mapped generic while detached, populate its
// scalar body, and only then insert it.
struct ElementwiseToLinalgGeneric : ConversionPattern {
  ElementwiseToLinalgGeneric(MLIRContext *context)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/100, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    StringRef dialect = op->getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math") ||
        !isFloatElementwise(op) || op->getNumResults() == 0)
      return failure();
    auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!result)
      return failure();
    const int64_t rank = result.getRank();
    for (Type type : op->getResultTypes())
      if (!isa<RankedTensorType>(type))
        return failure();

    AffineMap scalarMap = AffineMap::get(rank, /*symbolCount=*/0, {},
                                         rewriter.getContext());
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(operands.size() + op->getNumResults());
    for (Type type : op->getOperandTypes()) {
      if (isScalarElementwiseType(type))
        indexingMaps.push_back(scalarMap);
      else if (isa<RankedTensorType>(type))
        indexingMaps.push_back(identityMap);
      else
        return failure();
    }
    indexingMaps.append(op->getNumResults(), identityMap);

    SmallVector<Value> outputs;
    outputs.reserve(op->getNumResults());
    for (Type type : op->getResultTypes())
      outputs.push_back(emptyLike(rewriter, op->getLoc(), type));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    OperationState state(op->getLoc(), linalg::GenericOp::getOperationName());
    linalg::GenericOp::build(
        rewriter, state, op->getResultTypes(), operands, outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange bodyArgs) {
          SmallVector<Value> scalarArgs(bodyArgs.begin(),
                                        bodyArgs.begin() + operands.size());
          SmallVector<Type> scalarResultTypes;
          for (Type type : op->getResultTypes())
            scalarResultTypes.push_back(getElementTypeOrSelf(type));
          OperationState scalarState(bodyLoc, op->getName().getStringRef());
          scalarState.addOperands(scalarArgs);
          scalarState.addTypes(scalarResultTypes);
          scalarState.addAttributes(op->getAttrs());
          Operation *scalarOp = Operation::create(scalarState);
          bodyBuilder.insert(scalarOp);
          if (std::getenv("TTATLAS_TRACE_P3")) {
            OperationName yieldName(linalg::YieldOp::getOperationName(),
                                    bodyBuilder.getContext());
            llvm::errs() << "ttatlas P3 yield registered="
                         << yieldName.isRegistered() << " terminator="
                         << yieldName.hasTrait<OpTrait::IsTerminator>() << "\n";
          }
        });
    assert(state.regions.size() == 1 && "linalg.generic must have one region");
    // Create the operation only after the structured builder has populated the
    // detached body.  This avoids the runtime's eager verifier observing an
    // empty generic while still using its registered region/terminator ops.
    Operation *genericOperation = Operation::create(state);
    auto generic = cast<linalg::GenericOp>(genericOperation);
    rewriter.insert(genericOperation);
    copyTags(op, generic);
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

// A ReLU is the elementwise max of a tensor and zero.  Use linalg.max rather
// than linalg.generic: the stock Triton runtime verifies this named operation
// but does not expose a compatible generic-region implementation to plugins.
struct MaxNumToLinalgMax : OpConversionPattern<arith::MaxNumFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();
    Location loc = op.getLoc();
    Value output = emptyLike(rewriter, loc, op.getType());
    OperationState state(loc, linalg::MaxOp::getOperationName());
    SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs()};
    linalg::MaxOp::build(rewriter, state, operands, ValueRange{output});
    auto maxOperation = Operation::create(state);
    auto maximum = cast<linalg::MaxOp>(maxOperation);
    Region &region = maximum->getRegion(0);
    Block *body = region.empty() ? &region.emplaceBlock() : &region.front();
    operands.push_back(output);
    for (Value value : operands)
      body->addArgument(getElementTypeOrSelf(value.getType()), loc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(body);
      ImplicitLocOpBuilder regionBuilder(loc, rewriter);
      linalg::MaxOp::regionBuilder(regionBuilder, *body, {}, {});
    }
    rewriter.insert(maxOperation);
    copyTags(op, maximum);
    rewriter.replaceOp(op, maximum->getResults());
    return success();
  }
};

// Upstream's elementwise patterns reuse an operand as the generic's output
// when the body ignores it. Tile-and-fuse then duplicates that operand's
// producer (32 matmuls instead of 8 on a 64x64 MLP), so each such output
// becomes a fresh tensor.empty. A walk after the conversion, because
// upstream's patterns offer no hook to choose the output.
void freshOutputs(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  module.walk([&](linalg::GenericOp generic) {
    Block &body = generic.getRegion().front();
    const size_t inputCount = generic.getInputs().size();
    size_t outputIndex = 0;
    for (OpOperand &init : generic.getDpsInitsMutable()) {
      // payloadUsesValueFromOperand consults the generic's implicit
      // terminator trait.  Triton's runtime and our build headers disagree on
      // that trait, but the corresponding output block argument directly
      // expresses the condition we need here.
      if (!body.getArgument(inputCount + outputIndex).use_empty() ||
          init.get().getDefiningOp<tensor::EmptyOp>()) {
        ++outputIndex;
        continue;
      }
      rewriter.setInsertionPoint(generic);
      Value empty = emptyLike(rewriter, generic.getLoc(), init.get().getType());
      rewriter.modifyOpInPlace(generic, [&] { init.set(empty); });
      ++outputIndex;
    }
  });
}

// Upstream copies an arith op's attributes onto the scalar op inside the
// generic's body, so the ttatlas tags land there; they belong on the generic.
// A walk after the conversion, for the same reason as freshOutputs.
void hoistTags(ModuleOp module) {
  module.walk([](linalg::GenericOp generic) {
    for (Operation &op : generic.getRegion().front()) {
      if (isa<linalg::YieldOp>(op))
        continue;
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
  // The MLIR runtime loaded by Triton's Python extension does not expose the
  // same implicit-terminator trait as the headers used to build this plugin.
  // Running region DCE here consequently treats linalg.generic bodies as
  // dead and erases their scalar operation plus yield.  That leaves an
  // invalid empty generic before pass 4 can run.  Later canonicalization and
  // slot assignment already remove dead values, so keep the structured
  // regions intact at this boundary.
  (void)module;
}

// Rejects, before the conversion, the float compute this pass has no linalg
// form for, so it fails with a reason instead of slipping through: a
// tt.fp_to_fp between equal widths or narrowing with a rounding other than
// to nearest even (arith.truncf's), and any other tt op that makes a float
// tensor (tt.precise_sqrt, tt.extern_elementwise, ...). Upstream's elementwise
// pattern would wrap either, unconverted, in a linalg.generic, and the
// partial conversion leaves ops it has no target rule for. Pass 1 rejects all
// of these first; this keeps the rule when pass 3 runs alone. The stage-3
// contract check forbids the same set.
LogicalResult checkCompute(ModuleOp module) {
  bool failed = false;
  module.walk([&](Operation *op) {
    // A reduce is float compute by its operands, as pass 3's target judges
    // it; reducing to a scalar has no linalg.reduce form here (pass 1
    // accepts row reductions of a 2-D tile only).
    if (auto reduce = dyn_cast<ReduceOp>(op);
        reduce && llvm::any_of(reduce.getOperandTypes(), isFloatTensor) &&
        !llvm::all_of(reduce->getResultTypes(), isFloatTensor)) {
      tta::emitUnsupported(op, "reduces to a scalar; pass 3 converts row "
                               "reductions of a tile only");
      failed = true;
      return;
    }
    if (op->getName().getDialectNamespace() != "tt" ||
        !llvm::any_of(op->getResultTypes(), isFloatTensor))
      return;
    if (auto fpToFp = dyn_cast<FpToFpOp>(op)) {
      auto from =
          cast<FloatType>(getElementTypeOrSelf(fpToFp.getSrc().getType()));
      auto to = cast<FloatType>(getElementTypeOrSelf(fpToFp.getType()));
      std::optional<RoundingMode> rounding = fpToFp.getRounding();
      bool narrowing = to.getWidth() < from.getWidth();
      if (from.getWidth() == to.getWidth() ||
          (narrowing && rounding && *rounding != RoundingMode::RTNE)) {
        tta::emitUnsupported(op, "converts with no arith form (equal widths, "
                                 "or narrowing that isn't rtne)");
        failed = true;
      }
      return;
    }
    if (isa<LoadOp, DotOp, TransOp, BroadcastOp, SplatOp, ExpandDimsOp,
            ReduceOp>(op))
      return;
    tta::emitUnsupported(op, "pass 3 has no linalg form for this float op");
    failed = true;
  });
  return ::mlir::failure(failed);
}

struct ConvertTritonComputeToLinalg
    : impl::ConvertTritonComputeToLinalgBase<ConvertTritonComputeToLinalg> {
  void runOnOperation() override {
    const bool trace = std::getenv("TTATLAS_TRACE_P3") != nullptr;
    if (failed(checkCompute(getOperation())))
      return signalPassFailure();
    if (trace)
      llvm::errs() << "ttatlas P3: checked\n";
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
    target.addDynamicallyLegalOp<ReduceOp>([](ReduceOp op) {
      return !llvm::any_of(op.getOperandTypes(), isFloatTensor);
    });
    target.addDynamicallyLegalOp<ExpandDimsOp>([](ExpandDimsOp op) {
      return !isFloatTensor(op.getType()) || absorbedByBroadcasts(op);
    });
    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) { return !isFloatElementwise(op); });
    // Everything else stays as it is: loads, stores, pointers, address math
    // (checkCompute has rejected float compute without a pattern).
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<DotToMatmul, TransToTranspose, BroadcastToLinalg,
                 ExpandDimsToExpandShape, SplatToFill, ReduceToLinalg>(context);
    // Ahead of upstream's patterns, which would match tt.fp_to_fp too (it is
    // elementwise) and put it, unconverted, into a generic's body.
    patterns.add<FpToFpToArith>(context, /*benefit=*/2);
    patterns.add<ElementwiseToLinalgGeneric>(context);
    if (trace)
      llvm::errs() << "ttatlas P3: converting\n";
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
    if (trace)
      llvm::errs() << "ttatlas P3: converted\n";
    eraseDeadLeftovers(getOperation());
    if (trace)
      llvm::errs() << "ttatlas P3: dce\n";
    freshOutputs(getOperation());
    if (trace)
      llvm::errs() << "ttatlas P3: fresh outputs\n";
    hoistTags(getOperation());
    if (trace)
      llvm::errs() << "ttatlas P3: done\n";
    if (trace)
      getOperation()->print(llvm::errs());
  }
};

} // namespace

} // namespace mlir::triton
