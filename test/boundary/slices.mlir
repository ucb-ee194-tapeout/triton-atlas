// RUN: triton-atlas-opt %S/../fixtures/boundary-slices.ttir.mlir --ttatlas-assign-slots --convert-triton-to-atlas-virtual --ttatlas-verify-contract=stage=7 --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='tensor.'
// CHECK: func.func @sliced_copy() -> !atlas.virtual_state
// CHECK-SAME: atlas.input_dram_base = 2415919104 : i64
// CHECK-SAME: atlas.output_dram_base = 2415923200 : i64
// CHECK-SAME: ttatlas.kernel
// CHECK: atlas.virtual_start
// CHECK: atlas.virtual_input_bf16
// CHECK-SAME: index = 0
// CHECK: atlas.virtual_input_bf16
// CHECK-SAME: index = 1
// CHECK: atlas.virtual_output_bf16
// CHECK-SAME: index = 0
// CHECK: atlas.virtual_output_bf16
// CHECK-SAME: index = 1
// CHECK: return
