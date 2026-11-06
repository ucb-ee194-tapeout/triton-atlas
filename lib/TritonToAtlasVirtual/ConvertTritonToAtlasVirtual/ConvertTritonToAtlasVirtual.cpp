// Pass 7, linalg compute + TTIR memory -> Atlas virtual IR. No patterns yet:
// the base class generated from Passes.td does the pass bookkeeping;
// runOnOperation() runs one conversion with every pattern group, then attaches
// the kernel metadata. The pieces live next to it, one file each
// (TypeConversion.cpp, Dot.cpp, Elementwise.cpp, Reduce.cpp, Boundary.cpp,
// KernelMetadata.cpp; see Patterns.h), so adding a pattern never needs an edit
// here. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "Patterns.h"
#include "TritonToAtlasVirtual/Passes.h"

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

    // Partial conversion: an op with no pattern yet stays as it is, so the
    // pass runs on every kernel while the patterns are being written;
    // ttatlas-verify-contract{stage=7} reports what is left.
    ConversionTarget target(*context);
    target.addLegalDialect<atlas::AtlasDialect, func::FuncDialect>();
    ModuleOp module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))) ||
        failed(attachKernelMetadata(module)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
