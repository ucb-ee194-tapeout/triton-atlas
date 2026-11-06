// RUN: triton-atlas-opt %S/../boundary/copy.mlir --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.'

// Boundary conversion is independently usable while compute patterns and the
// complete kernel manifest producer are unfinished. Hand-tagged BF16 copy
// needs no compute/type mappings from another lane.
// CHECK: func.func @copy() -> !atlas.virtual_state
// CHECK: atlas.virtual_start
// CHECK: atlas.virtual_input_bf16
// CHECK: atlas.virtual_output_bf16
// CHECK: return
