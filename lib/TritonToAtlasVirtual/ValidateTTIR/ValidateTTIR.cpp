// Pass 1, ttatlas-validate-ttir. Contract: triton-atlas design agenda, page
// "Pass 1 · ttatlas-validate-ttir". It stops what no later pass lowers yet and
// records each load's and store's host dtype.
//
// The op inventory is a ConversionTarget (OpInventory.cpp), with conditions on
// some legal ops (Conditions.cpp). applyPartialConversion checks it: the first
// illegal op stops the pass with our reason and MLIR's "failed to legalize"
// error. Shape follows triton v3.9.0 third_party/amd/lib/TritonAMDGPUToLLVM.
#include "ValidateTTIR.h"

#include "TritonToAtlasVirtual/Passes.h"
// The dialects ValidateTTIR.td's dependentDialects names.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_VALIDATETTIR
#include "TritonToAtlasVirtual/Passes.h.inc"

namespace {

namespace tta = mlir::ttatlas;
using namespace validate;

// Any illegal op: reports why, then fails, so the conversion stops on it.
// Benefit 0, so a dropped op's EraseDropped runs instead.
struct RejectUnsupported : ConversionPattern {
  RejectUnsupported(MLIRContext *ctx, std::shared_ptr<const Reasons> reasons)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/0, ctx),
        reasons(std::move(reasons)) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &) const override {
    // Prints "atlas: unsupported <op>: "; the reason follows.
    auto emit = [&] { return tta::emitUnsupported(op, ""); };
    if (auto it = reasons->find(op->getName()); it != reasons->end())
      emit() << it->second;
    else if (succeeded(checkConditions(op, emit))) // else it reported why
      emit() << "not an op TTIR contains (not in pass 1's op inventory)";
    return failure();
  }

  std::shared_ptr<const Reasons> reasons;
};

// Recorded now because pass 5 retypes values; the driver reads it to convert
// at the boundary (e.g. an f32 output Atlas writes as BF16).
void recordHostDtypes(ModuleOp module) {
  module.walk([](LoadOp load) {
    load->setAttr(tta::kHostDtype, TypeAttr::get(pointee(load.getPtr())));
  });
  module.walk([](StoreOp store) {
    store->setAttr(tta::kHostDtype, TypeAttr::get(pointee(store.getPtr())));
  });
}

struct ValidateTTIR : impl::ValidateTTIRBase<ValidateTTIR> {
  // Builds the target and the patterns once per pass instance.
  LogicalResult initialize(MLIRContext *ctx) override {
    target = std::make_shared<ConversionTarget>(*ctx);
    auto reasons = std::make_shared<Reasons>();
    buildOpInventory(*target, *reasons);

    RewritePatternSet set(ctx);
    set.add<RejectUnsupported>(ctx, std::move(reasons));
    populateDropPatterns(set);
    patterns = std::move(set);
    return success();
  }

  void runOnOperation() override {
    ConversionConfig config;
    // Validation only: never fold an illegal op into something legal.
    config.foldingMode = DialectConversionFoldingMode::Never;
    if (failed(
            applyPartialConversion(getOperation(), *target, patterns, config)))
      return signalPassFailure();
    recordHostDtypes(getOperation());
  }

  // Shared by the clones the pass manager makes for each thread; read only.
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

} // namespace

} // namespace mlir::triton
