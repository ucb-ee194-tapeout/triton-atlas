// Pass 5, ttatlas-legalize-dtypes. Contract: triton-atlas design agenda, page
// "Pass 5 · ttatlas-legalize-dtypes". Atlas has no f32 datapath: the MXUs take
// FP8 E4M3 inputs and the VPU computes in BF16 (docs/dialect-reference.md:42-43
// and docs/hardware-status.md:45 @ 8ee175c), while Triton returns f32 from
// every tl.dot. So every f32 value becomes BF16, retyped in place, except a
// matmul chain's accumulator, which stays in the MXU; a readout marker after
// each chain's last matmul says where its f32 result becomes BF16. The
// retyping is a dialect conversion driven by a TypeConverter, as the page's
// "Upstream MLIR to use" asks.
#include "TritonToAtlasVirtual/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton {

#define GEN_PASS_DEF_LEGALIZEDTYPES
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;

// `type` with f32 replaced by BF16: a scalar, the elements of a tensor, the
// pointee of a pointer, nested. Other types are returned as they are.
Type toBf16(Type type) {
  if (type.isF32())
    return BFloat16Type::get(type.getContext());
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.clone(toBf16(tensor.getElementType()));
  if (auto pointer = dyn_cast<PointerType>(type))
    return PointerType::get(toBf16(pointer.getPointeeType()),
                            pointer.getAddressSpace());
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

// The matmul that accumulates onto `matmul` next, taking it as its output
// (the chain pass 3 linked); null if there is none.
linalg::MatmulOp nextInChain(linalg::MatmulOp matmul) {
  for (OpOperand &use : matmul->getUses())
    if (auto next = dyn_cast<linalg::MatmulOp>(use.getOwner());
        next && next.isDpsInit(&use))
      return next;
  return nullptr;
}

// Gives `user` its own copy of the op defining operand `index`, when others
// use it too (CSE merges identical constants and tensor.empty ops), so it can
// keep its type while the others' copy is retyped.
void privatize(Operation *user, unsigned index) {
  Operation *def = user->getOperand(index).getDefiningOp();
  if (!def || def->hasOneUse())
    return;
  OpBuilder builder(user);
  user->setOperand(index, builder.clone(*def)->getResult(0));
}

// Whether `generic` only passes its input through, so linalg's
// canonicalization erases it: one input, one result of the same type, all
// loops parallel, identity maps, and a body that yields its input.
bool isPassThrough(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic->getNumResults() != 1 ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType type) {
                      return type == utils::IteratorType::parallel;
                    }) ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return false;
  auto yield = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  return yield->getOperand(0) == generic.getBody()->getArgument(0) &&
         generic.getDpsInputs()[0].getType() == generic->getResult(0).getType();
}

// Moves the ttatlas.* tags of `from`, an op about to be erased in favour of
// `to`, onto the op that will define the replacing value: the op defining
// `to`, looking through pass-through generics that are erased too. Pass 4
// tags a stored value's defining op with ttatlas.pad, and pass 6 reads it
// there; when that op is a cast or a pass-through generic this pass erases,
// the tag must move with the value. Another tag already on the target wins.
// For ttatlas.pad it fails, with an error, rather than drop or spread it: when
// it has nowhere to go (no defining op, or a constant, which the dialect
// doesn't allow a pad on), when the target has a different pad, or, if
// `checkUses`, when the value it would land on has other users, which would
// then look padded too. (Uses aren't counted inside the dialect conversion,
// where the old ones are still present.)
LogicalResult moveTags(Operation *from, Value to, RewriterBase &rewriter,
                       bool checkUses) {
  bool onlyUser = to.hasOneUse();
  while (auto generic = to.getDefiningOp<linalg::GenericOp>()) {
    if (!isPassThrough(generic))
      break;
    to = generic.getDpsInputs()[0];
    onlyUser = onlyUser && to.hasOneUse();
  }
  Operation *target = to.getDefiningOp();
  for (NamedAttribute attr : from->getDiscardableAttrs()) {
    if (!attr.getName().strref().starts_with("ttatlas."))
      continue;
    bool pad = attr.getName() == tta::kPad;
    if (!target || isa<arith::ConstantOp>(target)) {
      if (pad)
        return tta::emitUnsupported(from, "ttatlas.pad can't move to the "
                                          "value that replaces this op");
      continue;
    }
    if (Attribute existing = target->getAttr(attr.getName())) {
      if (pad && existing != attr.getValue())
        return tta::emitUnsupported(from, "ttatlas.pad differs from the one "
                                          "on the value that replaces this "
                                          "op");
      continue;
    }
    if (pad && checkUses && !onlyUser)
      return tta::emitUnsupported(from, "ttatlas.pad would also apply to the "
                                        "other users of the value that "
                                        "replaces this op");
    rewriter.modifyOpInPlace(
        target, [&] { target->setAttr(attr.getName(), attr.getValue()); });
  }
  return success();
}

// After each chain's last matmul, one linalg.generic{arith.truncf} from f32 to
// BF16: the readout marker pass 7 maps to mxu_readout_bf16. Without it nothing
// marks that point once the BF16 casts are erased: in the MLP the only truncf
// is after the ReLU. Returns the marker.
linalg::GenericOp addReadout(linalg::MatmulOp last) {
  OpBuilder builder(last->getContext());
  builder.setInsertionPointAfter(last);
  Location loc = last.getLoc();
  auto type = cast<RankedTensorType>(last->getResult(0).getType());
  auto bf16 = cast<RankedTensorType>(toBf16(type));
  Value empty = tensor::EmptyOp::create(builder, loc, bf16.getShape(),
                                        bf16.getElementType());
  AffineMap identity = builder.getMultiDimIdentityMap(type.getRank());
  SmallVector<utils::IteratorType> iterators(type.getRank(),
                                             utils::IteratorType::parallel);
  auto marker = linalg::GenericOp::create(
      builder, loc, TypeRange{bf16}, ValueRange{last->getResult(0)},
      ValueRange{empty}, ArrayRef<AffineMap>{identity, identity}, iterators,
      [&](OpBuilder &body, Location bodyLoc, ValueRange args) {
        Value narrow = arith::TruncFOp::create(body, bodyLoc,
                                               bf16.getElementType(), args[0]);
        linalg::YieldOp::create(body, bodyLoc, narrow);
      });
  last->getResult(0).replaceAllUsesExcept(marker->getResult(0), marker);
  return marker;
}

// Finds every matmul chain (linalg.fill -> matmul -> matmul ...), adds its
// readout marker, and returns the ops that keep f32: the fill, its zero and
// its tensor.empty, the matmuls, and the marker (its input is f32). Fails, with
// an error, on an accumulator this can't keep in the MXU: a matmul that
// neither starts from a fill nor continues a chain (pass 2 gives every chain
// a zero start), or a chain value used outside its chain before its end.
FailureOr<llvm::SmallPtrSet<Operation *, 16>>
markAccumulators(ModuleOp module) {
  llvm::SmallPtrSet<Operation *, 16> keep;
  SmallVector<linalg::MatmulOp> starts;
  bool failed = false;
  module.walk([&](linalg::MatmulOp matmul) {
    Operation *init = matmul.getDpsInits()[0].getDefiningOp();
    if (isa_and_nonnull<linalg::FillOp>(init))
      starts.push_back(matmul);
    else if (!isa_and_nonnull<linalg::MatmulOp>(init)) {
      // Also what pass 2's experimental fold-bias-into-dot option leaves (a
      // bias as the accumulator). Making that real needs this pass to turn
      // the f32 bias into the BF16 tile Atlas's load_acc takes.
      tta::emitUnsupported(matmul, "the accumulator neither starts from "
                                   "linalg.fill nor continues a matmul chain");
      failed = true;
    }
  });
  for (linalg::MatmulOp start : starts) {
    auto fill = start.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!llvm::all_of(fill->getUsers(), [](Operation *user) {
          return isa<linalg::MatmulOp>(user);
        })) {
      tta::emitUnsupported(fill, "the accumulator's zero is used outside its "
                                 "matmul chains");
      failed = true;
    }
    // A fill several chains share (after CSE) is cloned per chain: each chain
    // has its own accumulator, and pass 7 resets one MXU accumulator per
    // fill.
    privatize(start, start.getDpsInitOperand(0)->getOperandNumber());
    fill = start.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    for (unsigned i = 0, e = fill->getNumOperands(); i < e; ++i)
      privatize(fill, i);
    keep.insert(fill);
    for (Value operand : fill->getOperands())
      if (Operation *def = operand.getDefiningOp())
        keep.insert(def);
    linalg::MatmulOp last = start;
    for (linalg::MatmulOp step = start; step; step = nextInChain(step)) {
      if (step != start && !last->hasOneUse()) {
        tta::emitUnsupported(last, "a partial sum of the matmul chain is used "
                                   "outside it; it stays in the MXU");
        failed = true;
      }
      keep.insert(step);
      last = step;
    }
    keep.insert(addReadout(last));
  }
  if (failed)
    return failure();
  return keep;
}

// f32 -> BF16 for scalars, tensor elements and pointees (toBf16); every other
// type converts to itself.
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
    if (failed(moveTags(op, adaptor.getIn(), rewriter, /*checkUses=*/false)))
      return failure();
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }
};

// Retypes every f32 value to BF16, except in the ops kept f32 (a matmul
// chain's fill, matmuls and readout marker, with their bodies): one partial
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
  // The kept ops' bodies (implicit for fill and matmul) stay f32 too.
  target.addDynamicallyLegalOp<linalg::FillOp, linalg::MatmulOp,
                               linalg::GenericOp>(legal);
  target.markOpRecursivelyLegal<linalg::FillOp, linalg::MatmulOp,
                                linalg::GenericOp>(
      [&](Operation *op) { return keep.contains(op); });

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
LogicalResult eraseIdentityGenerics(ModuleOp module) {
  // Their tags first move to the input's defining op (moveTags), since the
  // canonicalization pattern drops them with the generic.
  IRRewriter rewriter(module.getContext());
  // moveTags looks through chained pass-throughs, so a tag lands on the real
  // producer even when several of them are erased in a row.
  WalkResult moved = module.walk([&](linalg::GenericOp generic) {
    if (!isPassThrough(generic))
      return WalkResult::advance();
    return failed(moveTags(generic, generic.getDpsInputs()[0], rewriter,
                           /*checkUses=*/true))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  if (moved.wasInterrupted())
    return failure();
  RewritePatternSet patterns(module.getContext());
  linalg::GenericOp::getCanonicalizationPatterns(patterns, module.getContext());
  (void)applyPatternsGreedily(module, std::move(patterns));
  return success();
}

// The precision policy (module tag ttatlas.precision_policy, set from the
// backend's options; the first policy, fp8-bf16-v1, when absent) decides which
// sources this pass can retype. fp8-bf16-v1 takes FP8 and BF16 sources only:
// converting an f32 kernel changes its results, and each policy's effect must
// be measured first (the page's "Why"). Pass 1 rejects f32 sources too, with
// the kernel's source line; this is the check of the pass that would lower
// them, so the rule holds when pass 5 runs alone.
LogicalResult checkSources(ModuleOp module) {
  auto tag = module->getAttrOfType<StringAttr>(tta::kPrecisionPolicy);
  StringRef policy =
      tag ? tag.getValue() : StringRef(tta::kPrecisionPolicies[0]);
  bool failed = false;
  module.walk([&](LoadOp load) {
    Type source =
        cast<PointerType>(getElementTypeOrSelf(load.getPtr().getType()))
            .getPointeeType();
    if (source.isF32()) {
      tta::emitUnsupported(load, "an f32 source; the precision policy ")
          << policy << " takes FP8 and BF16 sources only";
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
    FailureOr<llvm::SmallPtrSet<Operation *, 16>> keep =
        markAccumulators(module);
    if (failed(keep) || failed(retype(module, *keep)) ||
        failed(eraseIdentityGenerics(module)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
