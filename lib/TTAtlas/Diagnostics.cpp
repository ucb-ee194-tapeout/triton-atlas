// The one rejection format, and the one legalization record, every pass uses
// (triton-atlas design agenda, "Frozen encodings (v1)", Error format).
#include "TTAtlas/TTAtlas.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ttatlas {

InFlightDiagnostic emitUnsupported(Operation *op, const llvm::Twine &reason) {
  return op->emitError() << "atlas: unsupported " << op->getName() << ": "
                         << reason;
}

// "file:line:col" of the first file location inside `loc` (through names and
// call sites), or MLIR's own printing when there is none.
static std::string describe(Location loc) {
  std::string text;
  loc->walk([&](Location inner) {
    if (auto file = dyn_cast<FileLineColLoc>(inner)) {
      text = (file.getFilename().getValue() + ":" +
              llvm::Twine(file.getLine()) + ":" + llvm::Twine(file.getColumn()))
                 .str();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (text.empty())
    llvm::raw_string_ostream(text) << loc;
  return text;
}

void recordLegalization(Operation *op, llvm::StringRef pass,
                        const llvm::Twine &what) {
  auto module =
      isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  if (!module)
    return;
  Builder builder(op->getContext());
  DictionaryAttr entry = builder.getDictionaryAttr({
      builder.getNamedAttr("pass", builder.getStringAttr(pass)),
      builder.getNamedAttr("what", builder.getStringAttr(what)),
      builder.getNamedAttr("loc",
                           builder.getStringAttr(describe(op->getLoc()))),
  });
  SmallVector<Attribute> entries;
  if (auto existing = module->getAttrOfType<ArrayAttr>(kLegalized)) {
    // A greedy rewrite can fire the same pattern again; record each
    // legalization once.
    if (llvm::is_contained(existing, entry))
      return;
    entries.append(existing.begin(), existing.end());
  }
  entries.push_back(entry);
  module->setAttr(kLegalized, builder.getArrayAttr(entries));
}

} // namespace mlir::ttatlas
