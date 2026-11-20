// The vector layout helper (see VectorLayouts.h).
#include "TritonToAtlasVirtual/VectorLayouts.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir::triton {
namespace {

using ttatlas::VectorLayout;

bool isVector(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getRank() == 1;
}

// The layout a 2-D generic's map asks of its vector operand: the loop the
// vector follows is rows (d0) or columns (d1).
std::optional<VectorLayout> layoutOfMap(AffineMap map) {
  if (map.getNumResults() != 1)
    return std::nullopt;
  auto dim = dyn_cast<AffineDimExpr>(map.getResult(0));
  if (!dim || dim.getPosition() > 1)
    return std::nullopt;
  return dim.getPosition() == 0 ? VectorLayout::rows : VectorLayout::cols;
}

} // namespace

VectorLayout VectorLayouts::of(Value value) const {
  return computed.lookup_or(value, VectorLayout::rows);
}

VectorLayout VectorLayouts::ofUse(OpOperand &use) const {
  return uses.lookup_or(&use, VectorLayout::rows);
}

VectorLayouts computeVectorLayouts(Operation *root,
                                   function_ref<bool(Value)> isLoaded) {
  VectorLayouts result;
  llvm::EquivalenceClasses<Value> classes;
  SmallVector<Value> values; // every computed vector, to read back its class
  SmallVector<std::pair<Value, VectorLayout>> demands;
  SmallVector<std::pair<Value, OpOperand *>> loadedUses; // reader, use

  DenseSet<Value> seen;
  auto add = [&](Value value) {
    if (seen.insert(value).second)
      values.push_back(value);
    return classes.getOrInsertLeaderValue(value);
  };

  root->walk([&](linalg::GenericOp generic) {
    bool reduces = llvm::is_contained(generic.getIteratorTypesArray(),
                                      utils::IteratorType::reduction);
    bool matrix = llvm::any_of(generic->getResultTypes(), [](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      return tensor && tensor.getRank() > 1;
    });
    if (matrix) {
      // A broadcast (rank-1 input, 2-D result) asks for the layout of its map.
      for (OpOperand *input : generic.getDpsInputOperands()) {
        if (!isVector(input->get().getType()))
          continue;
        auto layout = layoutOfMap(generic.getMatchingIndexingMap(input));
        if (!layout)
          continue;
        if (isLoaded(input->get())) {
          result.uses[input] = *layout;
        } else {
          add(input->get());
          demands.push_back({input->get(), *layout});
        }
      }
      return;
    }
    // A vector generic (an elementwise op, or the init and result of a row
    // reduction): its vectors share one layout, and a reduction's is rows.
    Value first = generic->getResult(0);
    if (!isVector(first.getType()))
      return;
    add(first);
    for (OpOperand &init : generic.getDpsInitsMutable())
      classes.unionSets(add(init.get()), first);
    for (OpOperand *input : generic.getDpsInputOperands()) {
      if (!isVector(input->get().getType()))
        continue;
      if (isLoaded(input->get()))
        loadedUses.push_back({first, input});
      else
        classes.unionSets(add(input->get()), first);
    }
    if (reduces)
      demands.push_back({first, VectorLayout::rows});
  });

  DenseMap<Value, VectorLayout> fixed; // by class leader
  for (auto [value, layout] : demands) {
    auto [it, inserted] =
        fixed.try_emplace(classes.getLeaderValue(value), layout);
    if (!inserted && it->second != layout)
      result.conflicts.push_back(value.getDefiningOp());
  }
  for (Value value : values)
    result.computed[value] =
        fixed.lookup_or(classes.getLeaderValue(value), VectorLayout::rows);
  for (auto [reader, use] : loadedUses)
    result.uses[use] = result.of(reader);
  return result;
}

} // namespace mlir::triton
