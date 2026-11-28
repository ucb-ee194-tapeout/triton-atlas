// Pass 4, step 4a: data tiling. Every linalg op is packed into 32x32 blocks
// with upstream linalg::pack, so the memory boundary becomes linalg.pack /
// linalg.unpack ops that canonicalization and tiling respect. Blocks smaller
// than a tile are padded with zero by the pack, so the lanes a padded reader
// could carry into a valid lane are protected here: a BF16 row reduction gets
// a clamp in front of it, a contraction's K padding is the pack's zero.
#include "TritonToAtlasVirtual/Passes.h"

#include "TTAtlas/AtlasTarget.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdlib>
#include <iterator>

namespace mlir::triton {

#define GEN_PASS_DEF_PACKBLOCKS
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

constexpr int64_t kTile = ttatlas::target::kTileSize;

// BF16 raw bits for finite-safe clamp bounds.  Using +inf/-inf as the
// in-block identities keeps min/max transparent without relying on NaN
// propagation differences between Atlas's VPU and npu_model.
constexpr uint16_t kClampHigh = 0x7F80;
constexpr uint16_t kClampLow = 0xFF80;
constexpr uint16_t kNegInf = 0xFF80;
constexpr uint16_t kPosInf = 0x7F80;

// Do not use Block::getTerminator here.  The plugin creates linalg regions
// with a yield operation, but the Triton wheel can carry a different
// SingleBlockImplicitTerminator TypeID.  Looking at the operation itself is
// equivalent for these regions and avoids an assertion in the host MLIR.
linalg::YieldOp getYield(Block &body) {
  if (body.empty())
    return {};
  Operation &last = body.back();
  if (last.getName().getStringRef() != linalg::YieldOp::getOperationName())
    return {};
  return cast<linalg::YieldOp>(&last);
}

// linalg transforms in the Triton wheel can clone a structured region without
// carrying over its implicit yield trait.  Restore the explicit yield from
// the final scalar producer before the next Atlas pass inspects the body.
bool repairYield(linalg::GenericOp op) {
  Block &body = *op.getBody();
  if (getYield(body))
    return true;
  if (body.empty() || op.getNumDpsInits() != 1)
    return false;
  Operation *producer = nullptr;
  for (Operation &inner : body) {
    if (inner.getName().getStringRef() == linalg::YieldOp::getOperationName())
      continue;
    if (inner.getNumResults() != 0)
      producer = &inner;
  }
  if (!producer || producer->getNumResults() != 1)
    return false;
  OpBuilder builder(op.getContext());
  builder.setInsertionPointToEnd(&body);
  builder.create<linalg::YieldOp>(producer->getLoc(), producer->getResult(0));
  return true;
}

// Preserve a scalar body across linalg::pack.  The transform in Triton's
// wheel can clone the generic region without its body when its linalg trait
// TypeID differs from the plugin's.  A detached snapshot lets us restore the
// exact scalar operation and its block-argument mapping after the transform.
void restoreBody(Operation *snapshot, linalg::GenericOp packed) {
  Region &sourceRegion = snapshot->getRegion(0);
  if (sourceRegion.empty())
    return;
  Block &source = sourceRegion.front();
  Block &destination = *packed.getBody();
  while (!destination.empty())
    destination.back().erase();
  IRMapping mapping;
  for (auto [from, to] : llvm::zip(source.getArguments(),
                                   destination.getArguments()))
    mapping.map(from, to);
  OpBuilder builder(packed.getContext());
  builder.setInsertionPointToEnd(&destination);
  for (Operation &inner : source) {
    if (inner.getName().getStringRef() == linalg::YieldOp::getOperationName())
      continue;
    builder.clone(inner, mapping);
  }
}

enum class ReaderKind { sum, max, min };

// The reader of a padded column dimension that `op` is, or nothing when it
// reduces no padded dimension. Contractions are not readers here: their K
// padding is zero. Sets `rejected` for a padded reduction with no rule.
std::optional<ReaderKind> paddedReader(linalg::GenericOp op, bool &rejected) {
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  bool padded = false;
  for (auto [range, iterator] : llvm::zip_equal(ranges, iterators))
    padded |= iterator == utils::IteratorType::reduction && range % kTile != 0;
  if (!padded || linalg::isaContractionOpInterface(op))
    return std::nullopt;
  SmallVector<Operation *> combiners;
  SmallVector<BlockArgument> accumulators(op.getRegionOutputArgs().begin(),
                                          op.getRegionOutputArgs().end());
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      !getElementTypeOrSelf(op.getDpsInputs()[0].getType()).isBF16() ||
      !matchReduction(accumulators, 0, combiners) || combiners.size() != 1) {
    rejected = true;
    return std::nullopt;
  }
  if (isa<arith::AddFOp>(combiners.front()))
    return ReaderKind::sum;
  if (isa<arith::MaxNumFOp, arith::MaximumFOp>(combiners.front()))
    return ReaderKind::max;
  if (isa<arith::MinNumFOp, arith::MinimumFOp>(combiners.front()))
    return ReaderKind::min;
  rejected = true;
  return std::nullopt;
}

// A tile grid of BF16 raw bits: `inside` for the lanes that map into the
// source block of `pack`, `outside` for the padded lanes.
Value blockConstant(OpBuilder &builder, Location loc, linalg::PackOp pack,
                    uint16_t inside, uint16_t outside) {
  auto type = cast<RankedTensorType>(pack.getDest().getType());
  ArrayRef<int64_t> block = pack.getSourceType().getShape();
  ArrayRef<int64_t> grid = type.getShape();
  ArrayRef<int64_t> positions = pack.getInnerDimsPos();
  SmallVector<uint16_t> bits;
  for (int64_t o0 = 0; o0 < grid[0]; ++o0)
    for (int64_t o1 = 0; o1 < grid[1]; ++o1)
      for (int64_t i0 = 0; i0 < kTile; ++i0)
        for (int64_t i1 = 0; i1 < kTile; ++i1) {
          int64_t outer[2] = {o0, o1}, inner[2] = {i0, i1}, at[2] = {};
          for (int k = 0; k < 2; ++k)
            at[positions[k]] = inner[k];
          bool in = true;
          for (int d = 0; d < 2; ++d)
            in &= outer[d] * kTile + at[d] < block[d];
          bits.push_back(in ? inside : outside);
        }
  auto attr = DenseElementsAttr::getFromRawBuffer(
      type, ArrayRef<char>(reinterpret_cast<char *>(bits.data()),
                           bits.size() * sizeof(uint16_t)));
  return arith::ConstantOp::create(builder, loc, type, attr);
}

// Puts a clamp between `reader`'s operand (the pack `pack`) and the reader:
// x' = max(min(x, high), low), a generic over the packed tile grid. Lanes
// inside the Triton block keep their bits; lanes outside become the reader's
// identity. A max reader needs only the min, a min reader only the max.
void clampReaderOperand(IRRewriter &rewriter, linalg::LinalgOp reader,
                        linalg::PackOp pack, ReaderKind kind) {
  Location loc = reader.getLoc();
  rewriter.setInsertionPoint(reader);
  Value x = pack.getResult();
  auto type = cast<RankedTensorType>(x.getType());
  bool hasMin = kind != ReaderKind::min, hasMax = kind != ReaderKind::max;
  uint16_t identity = kind == ReaderKind::max   ? kNegInf
                      : kind == ReaderKind::min ? kPosInf
                                                : 0;
  SmallVector<Value> inputs{x};
  if (hasMin)
    inputs.push_back(blockConstant(rewriter, loc, pack, kClampHigh,
                                   kind == ReaderKind::sum ? 0 : identity));
  if (hasMax)
    inputs.push_back(blockConstant(rewriter, loc, pack, kClampLow,
                                   kind == ReaderKind::sum ? 0 : identity));
  Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                        type.getElementType());
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(type.getRank());
  auto clamp = linalg::GenericOp::create(
      rewriter, loc, TypeRange{type}, inputs, ValueRange{empty},
      SmallVector<AffineMap>(inputs.size() + 1, identityMap),
      SmallVector<utils::IteratorType>(type.getRank(),
                                       utils::IteratorType::parallel),
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value value = args[0];
        unsigned next = 1;
        if (hasMin)
          value = arith::MinNumFOp::create(b, l, value, args[next++]);
        if (hasMax)
          value = arith::MaxNumFOp::create(b, l, value, args[next++]);
        linalg::YieldOp::create(b, l, value);
      });
  rewriter.modifyOpInPlace(
      reader, [&] { reader.getDpsInputOperand(0)->set(clamp.getResult(0)); });
}

// A transposed load (a pure permutation copy of a tt.load result) read by
// generics: the consumers read the load through the permuted indexing map
// instead, so the load is packed once, in the layout the reader wants
// (a transposed weight is then one pack of the load).
void foldLoadTransposes(IRRewriter &rewriter, ModuleOp module) {
  if (std::getenv("TTATLAS_PLUGIN_RUNTIME"))
    return;
  SmallVector<linalg::GenericOp> copies;
  module.walk([&](linalg::GenericOp op) {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
        !op.getInputs()[0].getDefiningOp<LoadOp>() ||
        !llvm::all_of(op.getIteratorTypesArray(), linalg::isParallelIterator))
      return;
    auto yield = getYield(*op.getBody());
    if (!yield)
      return;
    if (yield.getOperand(0) == op.getBody()->getArgument(0) &&
        op.getMatchingIndexingMap(op.getDpsInputOperand(0)).isPermutation())
      copies.push_back(op);
  });
  for (linalg::GenericOp copy : copies)
    for (OpOperand &use : llvm::make_early_inc_range(copy->getUses())) {
      auto consumer = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (!consumer || !linalg::areElementwiseOpsFusable(&use))
        continue;
      rewriter.setInsertionPoint(consumer);
      FailureOr<linalg::ElementwiseOpFusionResult> fused =
          linalg::fuseElementwiseOps(rewriter, &use);
      if (succeeded(fused))
        for (auto [old, replacement] : fused->replacements)
          rewriter.replaceAllUsesWith(old, replacement);
    }
}

struct PackBlocks : impl::PackBlocksBase<PackBlocks> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    const bool trace = std::getenv("TTATLAS_TRACE_P4") != nullptr;
    if (trace)
      llvm::errs() << "ttatlas P4: begin\n";
    IRRewriter rewriter(&getContext());
    auto reject = [&](Operation *op, const Twine &why) {
      ttatlas::emitUnsupported(op, why);
      signalPassFailure();
    };

    SmallVector<linalg::LinalgOp> named;
    module.walk([&](linalg::LinalgOp op) {
      if (!isa<linalg::GenericOp>(op))
        named.push_back(op);
    });
    if (trace)
      llvm::errs() << "ttatlas P4: named=" << named.size() << "\n";
    if (std::getenv("TTATLAS_TRACE_BODY")) {
      for (linalg::LinalgOp op : named)
        llvm::errs() << "ttatlas P4 named " << op->getName().getStringRef()
                     << "\n";
      module.walk([&](linalg::GenericOp op) {
        llvm::errs() << "ttatlas P4 pre-generalize generic body="
                     << std::distance(op.getBody()->begin(), op.getBody()->end())
                     << " args="
                     << op.getBody()->getNumArguments() << "\n";
        for (Operation &inner : *op.getBody())
          llvm::errs() << "  " << inner.getName().getStringRef() << "\n";
      });
    }
    for (linalg::LinalgOp op : named) {
      rewriter.setInsertionPoint(op);
      if (failed(linalg::generalizeNamedOp(rewriter, op)))
        return reject(op, "cannot generalize this linalg op");
    }

    if (std::getenv("TTATLAS_TRACE_BODY")) {
      module.walk([&](linalg::GenericOp op) {
        llvm::errs() << "ttatlas P4 post-generalize generic body="
                     << std::distance(op.getBody()->begin(), op.getBody()->end())
                     << " args="
                     << op.getBody()->getNumArguments() << "\n";
        for (Operation &inner : *op.getBody())
          llvm::errs() << "  " << inner.getName().getStringRef() << "\n";
      });
    }

    foldLoadTransposes(rewriter, module);
    if (trace)
      llvm::errs() << "ttatlas P4: folded transposes\n";

    SmallVector<linalg::GenericOp> generics;
    module.walk([&](linalg::GenericOp op) { generics.push_back(op); });
    for (linalg::GenericOp op : generics)
      if (!repairYield(op))
        return reject(op, "a linalg.generic has no recoverable yield");
    if (trace)
      llvm::errs() << "ttatlas P4: generics=" << generics.size() << "\n";
    for (linalg::GenericOp op : generics) {
      if (op.getNumLoops() == 0)
        continue;
      // A contraction's K padding is the pack's zero, which only holds when
      // the host wrote that zero: operands that are loads.
      bool paddedK = false;
      if (linalg::isaContractionOpInterface(op))
        for (auto [range, iterator] : llvm::zip_equal(
                 op.getStaticLoopRanges(), op.getIteratorTypesArray()))
          paddedK |=
              iterator == utils::IteratorType::reduction && range % kTile != 0;
      if (paddedK && !llvm::all_of(op.getDpsInputs(), [](Value operand) {
            return operand.getDefiningOp<LoadOp>();
          }))
        return reject(op, "a contraction with a padded K must read both "
                          "operands straight from loads");
      bool rejected = false;
      std::optional<ReaderKind> reader = paddedReader(op, rejected);
      if (rejected)
        return reject(op, "a padded reduction needs a single BF16 add, max or "
                          "min combiner over one input");

      rewriter.setInsertionPoint(op);
      Operation *snapshot =
          std::getenv("TTATLAS_PLUGIN_RUNTIME") ? op->clone() : nullptr;
      SmallVector<OpFoldResult> sizes(op.getNumLoops(),
                                      rewriter.getIndexAttr(kTile));
      FailureOr<linalg::PackResult> packed = linalg::pack(rewriter, op, sizes);
      if (failed(packed)) {
        if (snapshot)
          snapshot->destroy();
        return reject(op, "cannot pack this linalg op into 32x32 blocks");
      }
      if (auto packedGeneric = dyn_cast<linalg::GenericOp>(
              packed->packedLinalgOp.getOperation());
          packedGeneric) {
        if (!repairYield(packedGeneric) && snapshot)
          restoreBody(snapshot, packedGeneric);
        if (!repairYield(packedGeneric)) {
          if (snapshot)
            snapshot->destroy();
          return reject(packedGeneric,
                        "packing dropped the linalg.generic yield");
        }
      }
      if (snapshot)
        snapshot->destroy();
      if (llvm::any_of(packed->packOps, [](linalg::PackOp pack) {
            return !pack.getOuterDimsPerm().empty();
          }))
        return reject(packed->packedLinalgOp, "packing permuted the grid");
      if (reader)
        clampReaderOperand(rewriter, packed->packedLinalgOp,
                           packed->packOps.front(), *reader);
    }

    // A pack of an empty tensor is an empty tensor; a pack of a splat constant
    // is a splat constant (its pad lanes hold the splat value too).
    SmallVector<linalg::PackOp> packs;
    module.walk([&](linalg::PackOp pack) { packs.push_back(pack); });
    for (linalg::PackOp pack : packs) {
      Operation *source = pack.getSource().getDefiningOp();
      rewriter.setInsertionPoint(pack);
      if (isa_and_nonnull<tensor::EmptyOp>(source)) {
        auto type = pack.getDestType();
        rewriter.replaceOpWithNewOp<tensor::EmptyOp>(pack, type.getShape(),
                                                     type.getElementType());
      } else if (auto constant = dyn_cast_or_null<arith::ConstantOp>(source)) {
        auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
        if (!dense || !dense.isSplat())
          return reject(pack, "a pack of a constant that is not a splat");
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            pack, DenseElementsAttr::get(pack.getDestType(),
                                         dense.getSplatValue<Attribute>()));
      }
    }
  }
};

} // namespace
} // namespace mlir::triton
