// Passes that lower Triton IR (TTIR) to Atlas virtual ops (atlas-mlir's `atlas.virtual_*`).
// Layout follows Triton's AMD backend (triton v3.9.0 third_party/amd/include/TritonAMDGPUToLLVM/Passes.td).
// One file per pass under Passes/, so each lane edits only its own file. Run order: lib/TritonToAtlasVirtual/Pipeline.cpp.
#ifndef TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES
#define TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES

include "mlir/Pass/PassBase.td"

include "Passes/ValidateTTIR.td"                 // pass 1
include "Passes/Normalize.td"                    // pass 2
include "Passes/ConvertTritonComputeToLinalg.td" // pass 3
include "Passes/LegalizeShapes.td"               // pass 4
include "Passes/LegalizeDtypes.td"               // pass 5
include "Passes/AssignSlots.td"                  // pass 6
include "Passes/ConvertTritonToAtlasVirtual.td"  // pass 7
include "Passes/VerifyContract.td"               // checker used by tests, not in the pipeline

#endif // TRITON_ATLAS_TRITONTOATLASVIRTUAL_PASSES
