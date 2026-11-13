// The pieces of pass 7 (convert-triton-to-atlas-virtual), one file each, so
// several people can work in the pass without editing the same file. Private to
// this folder; ConvertTritonToAtlasVirtual.cpp runs them all.
#ifndef TRITON_ATLAS_CONVERTTRITONTOATLASVIRTUAL_PATTERNS_H
#define TRITON_ATLAS_CONVERTTRITONTOATLASVIRTUAL_PATTERNS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

// Tensor and pointer types to Atlas tile types (TypeConversion.cpp).
void populateTypeConversions(TypeConverter &typeConverter);
// tt.dot: one dot, trans + dot, K chains (Dot.cpp).
void populateDotPatterns(TypeConverter &typeConverter,
                         RewritePatternSet &patterns);
// Elementwise arith/math ops to VPU ops (Elementwise.cpp).
void populateElementwisePatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns);
// tt.reduce + expand_dims + broadcast to VPU row reductions (Reduce.cpp).
void populateReducePatterns(TypeConverter &typeConverter,
                            RewritePatternSet &patterns);
// The kernel boundary: tt.func to an argument-free func.func, slotted
// tt.load / tt.store to virtual_input_* / virtual_output_bf16 on the state
// token (Boundary.cpp).
void populateBoundaryPatterns(TypeConverter &typeConverter,
                              RewritePatternSet &patterns);
// Does the kernel still hold compute that no pattern has lowered yet (a linalg
// op, TTIR compute, float arith or math)? Then the boundary conversion leaves
// the whole function as it is: it would clone that compute into an Atlas
// function nothing can finish.
// Drops the linalg.transpose of each weight_fp8_nk tile before the conversion
// (Dot.cpp).
void eraseNKTransposes(ModuleOp module);
bool hasUnloweredCompute(Operation *function);
// After the conversion: the DRAM bases and ttatlas.kernel on the function
// (KernelMetadata.cpp).
LogicalResult attachKernelMetadata(ModuleOp module);

} // namespace mlir::triton

#endif // TRITON_ATLAS_CONVERTTRITONTOATLASVIRTUAL_PATTERNS_H
