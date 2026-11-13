// RUN: triton-atlas-opt %s | triton-atlas-opt | FileCheck %s

// Every ttatlas tag parses, verifies and prints back unchanged (encodings: triton-atlas design agenda,
// "Frozen encodings (v1)").

// CHECK: #map = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 * 32, 0)>
// CHECK: module attributes {ttatlas.dram_base = 2415919104 : i64, ttatlas.layout = #ttatlas<layout tile_major>, ttatlas.legalized = [{loc = "softmax.py:14:5", pass = "ttatlas-legalize-shapes", what = "mask -> padding"}], ttatlas.precision_policy = "fp8-bf16-v1"}
module attributes {ttatlas.dram_base = 2415919104 : i64, ttatlas.layout = #ttatlas<layout tile_major>, ttatlas.legalized = [{loc = "softmax.py:14:5", pass = "ttatlas-legalize-shapes", what = "mask -> padding"}], ttatlas.precision_policy = "fp8-bf16-v1"} {
  // CHECK: func.func @tags() attributes {ttatlas.kernel = "{\22version\22: 1}"}
  func.func @tags() attributes {ttatlas.kernel = "{\22version\22: 1}"} {
    // CHECK: ttatlas.host_dtype = f8E4M3FN
    // CHECK-SAME: ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = weight_fp8_kn, tile_origin = #map, logical_shape = [32, 32], host_dtype = f8E4M3FN>
    %a = tt.get_program_id x {ttatlas.host_dtype = f8E4M3FN, ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 * 32, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : i32
    // CHECK: ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>, ttatlas.mxu = 0 : i32
    %b = tt.get_program_id x {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>, ttatlas.mxu = 0 : i32} : i32
    // CHECK: ttatlas.pad = #ttatlas.pad<neutral = neg_inf, logical_shape = [32, 20]>, ttatlas.scale = 127 : i32
    %c = tt.get_program_id x {ttatlas.pad = #ttatlas.pad<neutral = neg_inf, logical_shape = [32, 20]>, ttatlas.scale = 127 : i32} : i32
    // CHECK: ttatlas.pad = #ttatlas.pad<neutral = one, logical_shape = [32, 20]>
    %d = tt.get_program_id x {ttatlas.pad = #ttatlas.pad<neutral = one, logical_shape = [32, 20]>} : i32
    // CHECK: ttatlas.pad = #ttatlas.pad<neutral = pos_inf, logical_shape = [32, 20]>
    %e = tt.get_program_id x {ttatlas.pad = #ttatlas.pad<neutral = pos_inf, logical_shape = [32, 20]>} : i32
    return
  }
}
