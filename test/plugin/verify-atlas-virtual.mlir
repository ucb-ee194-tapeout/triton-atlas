// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --verify-atlas-virtual-stream | FileCheck %s

// The plugin registers atlas-mlir's verifier, so our output can be checked as Atlas virtual IR: a well-formed stream
// passes, and a broken one fails at the op that breaks it. The streams are one layer of atlas-mlir@036ee09
// examples/handoff/virtual_mlp/00-atlas-virtual-ssa.mlir (Apache-2.0, THIRD_PARTY_LICENSES/atlas-mlir.txt).

// CHECK-LABEL: func.func @one_layer
// CHECK: atlas.virtual_mxu_matmul
// CHECK: atlas.virtual_output_bf16
func.func @one_layer() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415927296 : i64} {
  %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %io2, %w = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %h = "atlas.virtual_mxu_matmul"(%x, %w) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
  %y = "atlas.virtual_vpu_unary"(%h) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
  %io3 = "atlas.virtual_output_bf16"(%io2, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  return %io3 : !atlas.virtual_state
}

// -----

// The second input reuses %io0 instead of the state the first input returned.
func.func @state_reused() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415927296 : i64} {
  %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  // expected-error @+1 {{nonlinear virtual state chain}}
  %io2, %w = "atlas.virtual_input_fp8"(%io0) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %h = "atlas.virtual_mxu_matmul"(%x, %w) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
  %io3 = "atlas.virtual_output_bf16"(%io2, %h) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  return %io3 : !atlas.virtual_state
}
