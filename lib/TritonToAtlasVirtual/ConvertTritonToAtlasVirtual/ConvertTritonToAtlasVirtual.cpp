// Pass 7, linalg compute + TTIR memory -> Atlas virtual IR. No patterns yet:
// the base class generated from Passes.td does the pass bookkeeping;
// runOnOperation() runs one conversion with every pattern group, then attaches
// the kernel metadata. The pieces live next to it, one file each
// (TypeConversion.cpp, Dot.cpp, Elementwise.cpp, Reduce.cpp, Boundary.cpp,
// KernelMetadata.cpp; see Patterns.h), so adding a pattern never needs an edit
// here. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"
#include "Atlas/AtlasTypes.h"
#include "TritonToAtlasVirtual/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONTOATLASVIRTUAL
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

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
    target.addIllegalDialect("tt", "scf", "linalg", "tensor", "arith",
                             "math");
    target.addIllegalOp<FuncOp, LoadOp, StoreOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
      FunctionType signature = function.getFunctionType();
      return signature.getNumInputs() == 0 && signature.getNumResults() == 1 &&
             signature.getResult(0) ==
                 atlas::VirtualStateType::get(context);
    });
    ModuleOp module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))) ||
        failed(attachKernelMetadata(module)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
