// The one rejection format every pass uses: "atlas: unsupported <op>:
// <reason>".
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

} // namespace mlir::ttatlas
