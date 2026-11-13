// Pass 7, linalg compute + TTIR memory -> Atlas virtual IR. No patterns yet:
// the base class generated from Passes.td does the pass bookkeeping;
// runOnOperation() runs one conversion with every pattern group, then attaches
// the kernel metadata. The pieces live next to it, one file each
// (TypeConversion.cpp, Dot.cpp, Elementwise.cpp, Reduce.cpp, Boundary.cpp,
// KernelMetadata.cpp; see Patterns.h), so adding a pattern never needs an edit
// here. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTypes.h"
#include "Patterns.h"
#include "TritonToAtlasVirtual/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONTOATLASVIRTUAL
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

// A zero fill and a generic's destination tensor are temporary linalg
// scaffolding. Their users can disappear in a later conversion pattern, so
// discard the resulting dead chain to a fixed point.  This also removes its
// f32 zero: virtual IR permits scalar i1/i32 only.
static void eraseDeadTensorScaffolding(ModuleOp module) {
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      bool removable =
          isa<linalg::FillOp, tensor::EmptyOp>(op) ||
          (isa<arith::ConstantOp>(op) && op->getNumResults() == 1 &&
           isa<FloatType, RankedTensorType>(op->getResult(0).getType()));
      if (removable && llvm::all_of(op->getResults(), [](Value value) {
            return value.use_empty();
          }))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  } while (changed);
}

struct ConvertTritonToAtlasVirtual
    : impl::ConvertTritonToAtlasVirtualBase<ConvertTritonToAtlasVirtual> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TypeConverter typeConverter;
    populateTypeConversions(typeConverter);
    RewritePatternSet patterns(context);
    populateDotPatterns(typeConverter, patterns);
    populateElementwisePatterns(typeConverter, patterns);
    populateReducePatterns(typeConverter, patterns);
    populateBoundaryPatterns(typeConverter, patterns);

    // No TTIR or tensor compute operation may leak into the virtual stage.
    ConversionTarget target(*context);
    target.addLegalDialect<atlas::AtlasDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect("tt", "scf", "linalg", "tensor", "math");
    // These are short-lived structural operands of a matmul chain. The dot
    // pattern consumes the zero fill together with its tensor.empty seed;
    // legalizing children first must not reject that intermediate form.
    target.addDynamicallyLegalOp<tensor::EmptyOp>([](tensor::EmptyOp empty) {
      return llvm::all_of(empty->getUsers(), [](Operation *user) {
        return user->getName().getDialectNamespace() == "linalg";
      });
    });
    // A fill is legal while only matmul accumulators read it (several chains
    // can share one after CSE); it is swept when nothing reads it any more.
    target.addDynamicallyLegalOp<linalg::FillOp>([](linalg::FillOp fill) {
      return fill->getNumResults() == 1 &&
             llvm::all_of(fill->getResult(0).getUses(), [](OpOperand &use) {
               return use.getOwner()->getName().getStringRef() ==
                          "linalg.matmul" &&
                      use.getOperandNumber() == 2;
             });
    });
    // The region bodies are erased together with the enclosing linalg op.
    // Keep their linalg.yield syntax legal until that enclosing rewrite runs.
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>([](Operation *op) {
      Operation *parent = op->getParentOp();
      return parent && parent->getName().getDialectNamespace() == "linalg";
    });
    // Boundary.cpp initially builds the I/O state chain against tensor values,
    // then its output pattern replaces each boundary with the converted tile.
    target.addDynamicallyLegalOp<atlas::VirtualOutputBF16Op>(
        [](atlas::VirtualOutputBF16Op output) {
          return isa<atlas::VirtualStateType>(output.getOperand(0).getType()) &&
                 isa<atlas::VirtualBF16Type>(output.getOperand(1).getType());
        });
    // Scalar arithmetic is part of the Stage 7 contract; tensor arithmetic
    // must instead be rewritten to a virtual tile operation.
    target.addDynamicallyLegalDialect<arith::ArithDialect>([](Operation *op) {
      // A tensor constant read only by linalg ops (the zero of a ReLU, a mask
      // tile) is consumed by the pattern that lowers its reader and erased
      // with it.
      if (isa<arith::ConstantOp>(op) && op->getNumResults() == 1 &&
          isa<RankedTensorType>(op->getResult(0).getType()))
        return llvm::all_of(op->getUsers(), [](Operation *user) {
          return user->getName().getDialectNamespace() == "linalg";
        });
      return llvm::all_of(op->getOperandTypes(),
                          [](Type type) { return !isa<TensorType>(type); }) &&
             llvm::all_of(op->getResultTypes(),
                          [](Type type) { return !isa<TensorType>(type); });
    });
    target.addIllegalOp<FuncOp, LoadOp, StoreOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
      FunctionType signature = function.getFunctionType();
      return signature.getNumInputs() == 0 && signature.getNumResults() == 1 &&
             signature.getResult(0) == atlas::VirtualStateType::get(context);
    });
    ModuleOp module = getOperation();
    eraseNKTransposes(module);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    eraseDeadTensorScaffolding(module);
    if (failed(attachKernelMetadata(module)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
