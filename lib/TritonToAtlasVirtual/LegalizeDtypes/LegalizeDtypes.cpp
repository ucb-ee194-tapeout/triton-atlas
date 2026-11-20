// Pass 5, ttatlas-legalize-dtypes. Atlas has no f32 datapath: the MXUs take
// FP8 E4M3 inputs and the VPU computes in BF16 (docs/dialect-reference.md:42-43
// and docs/hardware-status.md:45 @ 8ee175c), while Triton returns f32 from
// every tl.dot. So every f32 value becomes BF16, retyped in place, matmul
// accumulators included. The retyping is a dialect conversion driven by a
// TypeConverter, the MLIR way.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZEDTYPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;

// `type` with f32 replaced by BF16: a scalar or the elements of a tensor.
// Other types, pointers included, are returned as they are: loads and stores
// keep the host's pointer types.
Type toBf16(Type type) {
  if (type.isF32())
    return BFloat16Type::get(type.getContext());
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.clone(toBf16(tensor.getElementType()));
  return type;
}

// An f32 constant's value as BF16, rounded to nearest even; null if it isn't
// f32.
Attribute toBf16(Attribute value) {
  auto round = [](APFloat number) {
    bool lost;
    number.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &lost);
    return number;
  };
  if (auto number = dyn_cast<FloatAttr>(value);
      number && number.getType().isF32())
    return FloatAttr::get(toBf16(number.getType()), round(number.getValue()));
  if (auto dense = dyn_cast<DenseFPElementsAttr>(value);
      dense && dense.getElementType().isF32())
    return dense.mapValues(
        toBf16(dense.getElementType()),
        [&](const APFloat &number) { return round(number).bitcastToAPInt(); });
  return nullptr;
}

// A store keeps its host pointer type, so a value stored as f32 goes through an
// explicit cast: the value is rounded to BF16 (that truncf is erased once the
// value is BF16) and extended back by an extf, which keeps f32 and is added to
// `keep`.
void castStoredValues(ModuleOp module,
                      llvm::SmallPtrSet<Operation *, 16> &keep) {
  module.walk([&](StoreOp store) {
    Type type = store.getValue().getType();
    if (toBf16(type) == type)
      return;
    OpBuilder builder(store);
    Value narrow = arith::TruncFOp::create(builder, store.getLoc(),
                                           toBf16(type), store.getValue());
    auto wide = arith::ExtFOp::create(builder, store.getLoc(), type, narrow);
    store.getValueMutable().assign(wide);
    keep.insert(wide);
  });
}

// f32 -> BF16 for scalars and tensor elements (toBf16); every other type
// converts to itself.
struct Bf16TypeConverter : TypeConverter {
  Bf16TypeConverter() {
    addConversion([](Type type) { return toBf16(type); });
  }
};

// The kernel signature's conversion: like Bf16TypeConverter, except that a
// scalar float argument keeps its type. It is the host's value as the
// launcher passes it, and pass 1 accepts one only unused, so retyping it would
// change the kernel's interface for nothing.
struct SignatureTypeConverter : TypeConverter {
  SignatureTypeConverter() {
    addConversion(
        [](Type type) { return isa<FloatType>(type) ? type : toBf16(type); });
  }
};

// Whether `op` already has only Atlas types: no f32 in its results, operands,
// region arguments or, for a function, its signature.
bool hasAtlasTypes(Operation *op, const TypeConverter &converter) {
  if (auto func = dyn_cast<FuncOp>(op)) {
    SignatureTypeConverter signature;
    return signature.isSignatureLegal(func.getFunctionType()) &&
           signature.isLegal(&func.getBody());
  }
  return converter.isLegal(op) &&
         llvm::all_of(op->getRegions(), [&](Region &region) {
           return converter.isLegal(&region);
         });
}

// Any op with f32 types: rebuilt with converted result types and the
// converted operands, its regions moved over and their block arguments
// converted (the ops inside are converted in turn). Upstream's
// convertOpResultTypes does the same for ops without regions; linalg ops have
// regions, so this pattern covers both.
struct RetypeOp : ConversionPattern {
  RetypeOp(const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Functions go through the signature pattern, constants through
    // RetypeConstant: rebuilt here, an f32 constant would keep its f32 value
    // under a BF16 type, so one RetypeConstant can't round fails to legalize
    // instead.
    if (isa<FuncOp, ModuleOp, arith::ConstantOp>(op))
      return failure();
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         types, op->getAttrs(), op->getSuccessors());
    state.propertiesAttr = op->getPropertiesAsAttribute();
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();
    Operation *retyped = rewriter.create(state);
    for (auto [from, to] : llvm::zip(op->getRegions(), retyped->getRegions())) {
      rewriter.inlineRegionBefore(from, to, to.end());
      if (failed(rewriter.convertRegionTypes(&to, *getTypeConverter())))
        return failure();
    }
    rewriter.replaceOp(op, retyped->getResults());
    return success();
  }
};

// An f32 constant: the same value rounded to BF16 (nearest even).
struct RetypeConstant : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Attribute value = toBf16(op.getValue());
    if (!value)
      return failure();
    auto retyped = rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, cast<TypedAttr>(value));
    for (NamedAttribute attr : op->getDiscardableAttrs())
      retyped->setAttr(attr.getName(), attr.getValue());
    return success();
  }
};

// An extf or truncf between BF16 and f32 becomes BF16 -> BF16, an identity:
// replaced by its input (the MLP's bias extf and its final truncf).
template <typename CastOp>
struct EraseIdentityCast : OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (type != adaptor.getIn().getType())
      return failure();
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }
};

// Retypes every f32 value to BF16, except the casts in `keep`: one partial
// conversion with a TypeConverter, MLIR's mechanism for type changes. tt.func's
// signature goes through upstream's FunctionOpInterface pattern. Upstream has
// no patterns that retype linalg bodies, tt ops or constants; RetypeOp and
// RetypeConstant are those.
LogicalResult retype(ModuleOp module,
                     const llvm::SmallPtrSet<Operation *, 16> &keep) {
  MLIRContext *ctx = module.getContext();
  Bf16TypeConverter converter;
  ConversionTarget target(*ctx);
  auto legal = [&](Operation *op) {
    return keep.contains(op) || hasAtlasTypes(op, converter);
  };
  target.markUnknownOpDynamicallyLegal(legal);
  // A store's value matches its host pointer type (castStoredValues).
  target.addLegalOp<StoreOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<RetypeOp>(converter, ctx);
  patterns.add<RetypeConstant, EraseIdentityCast<arith::ExtFOp>,
               EraseIdentityCast<arith::TruncFOp>>(converter, ctx,
                                                   /*benefit=*/2);
  static const SignatureTypeConverter signature;
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, signature);
  return applyPartialConversion(module, target, std::move(patterns));
}

// A linalg.generic whose body only yields its input is left where a cast was
// erased; linalg's own canonicalization patterns erase it. The greedy driver
// also folds and erases dead ops elsewhere in the module, which is harmless:
// the pipeline runs canonicalize right after this pass anyway.
void eraseIdentityGenerics(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  linalg::GenericOp::getCanonicalizationPatterns(patterns, module.getContext());
  (void)applyPatternsGreedily(module, std::move(patterns));
}

// This pass retypes FP8 and BF16 sources only: converting an f32 kernel
// changes its results. Pass 1 rejects f32 sources too, with the kernel's source
// line; this is the check of the pass that would lower them, so the rule holds
// when pass 5 runs alone.
LogicalResult checkSources(ModuleOp module) {
  bool failed = false;
  module.walk([&](LoadOp load) {
    Type source =
        cast<PointerType>(getElementTypeOrSelf(load.getPtr().getType()))
            .getPointeeType();
    if (source.isF32()) {
      tta::emitUnsupported(load, "f32 inputs are not supported yet; use fp8 or "
                                 "bf16");
      failed = true;
    }
  });
  return ::mlir::failure(failed);
}

struct LegalizeDtypes : impl::LegalizeDtypesBase<LegalizeDtypes> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(checkSources(module)))
      return signalPassFailure();
    llvm::SmallPtrSet<Operation *, 16> keep;
    castStoredValues(module, keep);
    if (failed(retype(module, keep)))
      return signalPassFailure();
    eraseIdentityGenerics(module);
  }
};

} // namespace

} // namespace mlir::triton
