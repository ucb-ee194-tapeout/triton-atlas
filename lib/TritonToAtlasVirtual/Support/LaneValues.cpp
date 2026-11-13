// Per-lane evaluation of compile-time offset and mask trees (see LaneValues.h).
//
// Derived from pass 6's integerAt/pointerAt (AssignSlots.cpp on
// the pass 6 branch (boundary-metadata), commit 9d1c71b: the supported ops,
// base-argument + offset pointers) and pass 4's collectMaskBounds
// (LegalizeShapes.cpp). Unlike those, this evaluates whole tensors at once and
// keeps the axis that expand_dims and broadcast move.
#include "TritonToAtlasVirtual/LaneValues.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton {
namespace {

using Lanes = SmallVector<int64_t>;

struct Evaluator {
  DenseMap<Value, std::optional<LaneValues>> memo;
  unsigned depth = 0;

  static std::optional<SmallVector<int64_t>> shapeOf(Type type) {
    SmallVector<int64_t> shape;
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      if (!tensor.hasStaticShape())
        return std::nullopt;
      shape.assign(tensor.getShape().begin(), tensor.getShape().end());
    }
    return shape;
  }
  static int64_t count(ArrayRef<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  }
  static unsigned widthOf(Type type) {
    Type elem = getElementTypeOrSelf(type);
    return elem.isIndex() ? 64 : elem.getIntOrFloatBitWidth();
  }
  // Normalize to the canonical form: i1 as 0/1, wider integers sign-extended
  // from their width.
  static int64_t wrap(int64_t v, unsigned width) {
    if (width >= 64)
      return v;
    if (width == 1)
      return v & 1;
    return llvm::SignExtend64(static_cast<uint64_t>(v), width);
  }
  static uint64_t unsignedOf(int64_t v, unsigned width) {
    return width >= 64 ? static_cast<uint64_t>(v)
                       : static_cast<uint64_t>(v) & ((1ull << width) - 1);
  }

  std::optional<LaneValues> eval(Value value) {
    auto it = memo.find(value);
    if (it != memo.end())
      return it->second;
    if (++depth > 256) {
      --depth;
      return std::nullopt;
    }
    auto result = compute(value);
    --depth;
    memo[value] = result;
    return result;
  }

  // Map each lane of `shape` to the source lane for broadcast/expand_dims:
  // source dims of size 1 read index 0.
  static LaneValues broadcastTo(const LaneValues &src, ArrayRef<int64_t> shape,
                                ArrayRef<int64_t> srcShapeAligned) {
    LaneValues out;
    out.shape.assign(shape.begin(), shape.end());
    int64_t total = count(shape);
    out.values.reserve(total);
    unsigned rank = shape.size();
    SmallVector<int64_t> srcStride(rank, 0);
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
      srcStride[i] = srcShapeAligned[i] == 1 ? 0 : stride;
      stride *= srcShapeAligned[i];
    }
    SmallVector<int64_t> idx(rank, 0);
    for (int64_t lane = 0; lane < total; ++lane) {
      int64_t s = 0;
      for (unsigned d = 0; d < rank; ++d)
        s += idx[d] * srcStride[d];
      out.values.push_back(src.values[s]);
      for (int d = rank - 1; d >= 0; --d) {
        if (++idx[d] < shape[d])
          break;
        idx[d] = 0;
      }
    }
    return out;
  }

  std::optional<LaneValues> compute(Value value) {
    Type type = value.getType();
    if (!getElementTypeOrSelf(type).isIntOrIndex())
      return std::nullopt;
    auto shape = shapeOf(type);
    if (!shape || count(*shape) > kMaxEvaluatedLanes)
      return std::nullopt;
    unsigned width = widthOf(type);
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    int64_t lanes = count(*shape);
    LaneValues out;
    out.shape = *shape;

    // Constants: scalar, splat or non-splat dense.
    if (isa<arith::ConstantOp>(op)) {
      APInt scalar;
      if (matchPattern(value, m_ConstantInt(&scalar))) {
        out.values.assign(lanes, wrap(scalar.getSExtValue(), width));
        return out;
      }
      DenseIntElementsAttr dense;
      if (matchPattern(value, m_Constant(&dense))) {
        for (const APInt &v : dense.getValues<APInt>())
          out.values.push_back(width == 1 ? (v.getZExtValue() & 1)
                                          : wrap(v.getSExtValue(), width));
        if (static_cast<int64_t>(out.values.size()) == lanes)
          return out;
      }
      return std::nullopt;
    }
    if (auto range = dyn_cast<MakeRangeOp>(op)) {
      for (int64_t i = 0; i < lanes; ++i)
        out.values.push_back(wrap(range.getStart() + i, width));
      return out;
    }
    if (auto splat = dyn_cast<SplatOp>(op)) {
      auto src = eval(splat.getSrc());
      if (!src)
        return std::nullopt;
      out.values.assign(lanes, src->values[0]);
      return out;
    }
    if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
      auto src = eval(expand.getSrc());
      if (!src)
        return std::nullopt;
      // Same lanes in the same order; the new axis has size 1.
      out.values = src->values;
      return out;
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
      auto src = eval(broadcast.getSrc());
      if (!src || src->shape.size() != shape->size())
        return std::nullopt;
      return broadcastTo(*src, *shape, src->shape);
    }
    if (auto reshape = dyn_cast<ReshapeOp>(op)) {
      auto src = eval(reshape.getSrc());
      if (!src || count(src->shape) != lanes)
        return std::nullopt;
      out.values = src->values; // row-major order is unchanged
      return out;
    }

    // Casts.
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>(op)) {
      auto src = eval(op->getOperand(0));
      if (!src)
        return std::nullopt;
      unsigned srcWidth = widthOf(op->getOperand(0).getType());
      for (int64_t v : src->values) {
        int64_t r = isa<arith::ExtUIOp, arith::IndexCastUIOp>(op)
                        ? static_cast<int64_t>(unsignedOf(v, srcWidth))
                        : v;
        out.values.push_back(wrap(r, width));
      }
      return out;
    }

    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      auto cond = eval(select.getCondition());
      auto t = eval(select.getTrueValue());
      auto f = eval(select.getFalseValue());
      if (!cond || !t || !f)
        return std::nullopt;
      // A scalar condition may select between tensors.
      for (int64_t i = 0; i < lanes; ++i) {
        int64_t c = cond->values[cond->values.size() == 1 ? 0 : i];
        out.values.push_back(c ? t->values[i] : f->values[i]);
      }
      return out;
    }

    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      auto lhs = eval(cmp.getLhs());
      auto rhs = eval(cmp.getRhs());
      if (!lhs || !rhs || lhs->values.size() != rhs->values.size())
        return std::nullopt;
      unsigned w = widthOf(cmp.getLhs().getType());
      for (int64_t i = 0; i < lanes; ++i) {
        int64_t a = lhs->values[i], b = rhs->values[i];
        uint64_t ua = unsignedOf(a, w), ub = unsignedOf(b, w);
        bool r = false;
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          r = a == b;
          break;
        case arith::CmpIPredicate::ne:
          r = a != b;
          break;
        case arith::CmpIPredicate::slt:
          r = a < b;
          break;
        case arith::CmpIPredicate::sle:
          r = a <= b;
          break;
        case arith::CmpIPredicate::sgt:
          r = a > b;
          break;
        case arith::CmpIPredicate::sge:
          r = a >= b;
          break;
        case arith::CmpIPredicate::ult:
          r = ua < ub;
          break;
        case arith::CmpIPredicate::ule:
          r = ua <= ub;
          break;
        case arith::CmpIPredicate::ugt:
          r = ua > ub;
          break;
        case arith::CmpIPredicate::uge:
          r = ua >= ub;
          break;
        }
        out.values.push_back(r);
      }
      return out;
    }

    // Binary integer arithmetic. Wraps at the result width (the flags nsw/nuw
    // only make overflow poison).
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
            arith::OrIOp, arith::XOrIOp, arith::MaxSIOp, arith::MinSIOp,
            arith::ShLIOp, arith::ShRSIOp>(op)) {
      auto lhs = eval(op->getOperand(0));
      auto rhs = eval(op->getOperand(1));
      if (!lhs || !rhs || lhs->values.size() != rhs->values.size())
        return std::nullopt;
      for (int64_t i = 0; i < lanes; ++i) {
        int64_t a = lhs->values[i], b = rhs->values[i];
        uint64_t ua = unsignedOf(a, width), ub = unsignedOf(b, width);
        int64_t r = 0;
        if (isa<arith::AddIOp>(op))
          r = static_cast<int64_t>(ua + ub);
        else if (isa<arith::SubIOp>(op))
          r = static_cast<int64_t>(ua - ub);
        else if (isa<arith::MulIOp>(op))
          r = static_cast<int64_t>(ua * ub);
        else if (isa<arith::AndIOp>(op))
          r = a & b;
        else if (isa<arith::OrIOp>(op))
          r = a | b;
        else if (isa<arith::XOrIOp>(op))
          r = a ^ b;
        else if (isa<arith::MaxSIOp>(op))
          r = std::max(a, b);
        else if (isa<arith::MinSIOp>(op))
          r = std::min(a, b);
        else if (isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
                     arith::RemUIOp>(op)) {
          if (b == 0)
            return std::nullopt;
          bool isSigned = isa<arith::DivSIOp, arith::RemSIOp>(op);
          bool isDiv = isa<arith::DivSIOp, arith::DivUIOp>(op);
          if (isSigned) {
            if (b == -1 && a == INT64_MIN)
              return std::nullopt;
            r = isDiv ? a / b : a % b;
          } else {
            r = static_cast<int64_t>(isDiv ? ua / ub : ua % ub);
          }
        } else { // shifts
          if (ub >= width)
            return std::nullopt;
          r = isa<arith::ShLIOp>(op) ? static_cast<int64_t>(ua << ub)
                                     : (width == 1 ? a : a >> ub);
        }
        out.values.push_back(wrap(r, width));
      }
      return out;
    }
    return std::nullopt;
  }
};

std::optional<PointerLanes> pointerOf(Value value, Evaluator &ev,
                                      unsigned depth) {
  if (depth > 256)
    return std::nullopt;
  Type elem = getElementTypeOrSelf(value.getType());
  if (!isa<PointerType>(elem))
    return std::nullopt;
  auto shape = Evaluator::shapeOf(value.getType());
  if (!shape || Evaluator::count(*shape) > kMaxEvaluatedLanes)
    return std::nullopt;
  int64_t lanes = Evaluator::count(*shape);

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (!isa<PointerType>(arg.getType()))
      return std::nullopt;
    return PointerLanes{arg, LaneValues{{}, {0}}};
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto splat = dyn_cast<SplatOp>(op)) {
    auto src = pointerOf(splat.getSrc(), ev, depth + 1);
    if (!src)
      return std::nullopt;
    return PointerLanes{
        src->base, LaneValues{*shape, Lanes(lanes, src->offsets.values[0])}};
  }
  if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
    auto src = pointerOf(expand.getSrc(), ev, depth + 1);
    if (!src)
      return std::nullopt;
    src->offsets.shape = *shape;
    return src;
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
    auto src = pointerOf(broadcast.getSrc(), ev, depth + 1);
    if (!src || src->offsets.shape.size() != shape->size())
      return std::nullopt;
    return PointerLanes{src->base, Evaluator::broadcastTo(src->offsets, *shape,
                                                          src->offsets.shape)};
  }
  if (auto add = dyn_cast<AddPtrOp>(op)) {
    auto base = pointerOf(add.getPtr(), ev, depth + 1);
    auto offset = ev.eval(add.getOffset());
    if (!base || !offset)
      return std::nullopt;
    // Operands may be a scalar pointer plus a tensor offset (not valid TTIR) or
    // matching shapes; accept equal lane counts or a single-lane side.
    PointerLanes out{base->base, LaneValues{*shape, {}}};
    for (int64_t i = 0; i < lanes; ++i) {
      int64_t b =
          base->offsets.values[base->offsets.values.size() == 1 ? 0 : i];
      int64_t o = offset->values[offset->values.size() == 1 ? 0 : i];
      out.offsets.values.push_back(b + o);
    }
    return out;
  }
  return std::nullopt;
}

} // namespace

std::optional<LaneValues> evaluateLanes(Value value) {
  Evaluator ev;
  return ev.eval(value);
}

std::optional<PointerLanes> evaluatePointer(Value value) {
  Evaluator ev;
  return pointerOf(value, ev, 0);
}

std::optional<SmallVector<int64_t>> maskBounds(Value mask) {
  if (!getElementTypeOrSelf(mask.getType()).isInteger(1))
    return std::nullopt;
  auto lanes = evaluateLanes(mask);
  if (!lanes)
    return std::nullopt;
  unsigned rank = lanes->shape.size();
  // The extent of the true lanes along each axis, found from the true lane with
  // the largest index per axis.
  SmallVector<int64_t> bounds(rank, 0);
  SmallVector<int64_t> idx(rank, 0);
  for (int64_t v : lanes->values) {
    if (v)
      for (unsigned d = 0; d < rank; ++d)
        bounds[d] = std::max(bounds[d], idx[d] + 1);
    for (int d = rank - 1; d >= 0; --d) {
      if (++idx[d] < lanes->shape[d])
        break;
      idx[d] = 0;
    }
  }
  // The true lanes are the leading box exactly when the box holds as many lanes
  // as there are true lanes and every lane inside it is true.
  int64_t boxLanes = 1, trueLanes = 0;
  for (unsigned d = 0; d < rank; ++d)
    boxLanes *= bounds[d];
  for (int64_t v : lanes->values)
    trueLanes += v != 0;
  if (trueLanes == 0)
    return SmallVector<int64_t>(rank, 0);
  if (boxLanes != trueLanes)
    return std::nullopt;
  std::fill(idx.begin(), idx.end(), 0);
  for (int64_t v : lanes->values) {
    bool inside = true;
    for (unsigned d = 0; d < rank; ++d)
      inside &= idx[d] < bounds[d];
    if (inside != (v != 0))
      return std::nullopt;
    for (int d = rank - 1; d >= 0; --d) {
      if (++idx[d] < lanes->shape[d])
        break;
      idx[d] = 0;
    }
  }
  return bounds;
}

} // namespace mlir::triton
