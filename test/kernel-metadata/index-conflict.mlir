// RUN: not triton-atlas-opt %s --convert-triton-to-atlas-virtual 2>&1 | FileCheck %s
// CHECK: virtual boundary index disagrees with its preserved #ttatlas.slot
#map = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>
module {
  func.func @copy() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415921152 : i64} {
    %0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %next, %value = "atlas.virtual_input_bf16"(%0) <{index = 0 : i32}> {ttatlas.slot = #ttatlas.slot<index = 1, arg = 0, role = constant_bf16, tile_origin = #map, logical_shape = [32, 32], host_dtype = bf16>} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %1 = "atlas.virtual_output_bf16"(%next, %value) <{index = 0 : i32}> {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = #map, logical_shape = [32, 32], host_dtype = f32>} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %1 : !atlas.virtual_state
  }
}
