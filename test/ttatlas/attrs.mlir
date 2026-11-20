// RUN: triton-atlas-opt %s | triton-atlas-opt | FileCheck %s

// Every ttatlas tag parses, verifies and prints back unchanged.

// CHECK: module attributes {ttatlas.dram_base = 2415919104 : i64, ttatlas.layout = #ttatlas<layout tile_major>, ttatlas.precision_policy = "fp8-bf16-v1"}
module attributes {ttatlas.dram_base = 2415919104 : i64, ttatlas.layout = #ttatlas<layout tile_major>, ttatlas.precision_policy = "fp8-bf16-v1"} {
  // CHECK: func.func @tags() attributes {ttatlas.kernel = "{\22version\22: 2}"}
  func.func @tags() attributes {ttatlas.kernel = "{\22version\22: 2}"} {
    // CHECK: ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = weight_fp8_kn, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
    %a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = weight_fp8_kn, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32
    // CHECK: ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>
    %b = tt.get_program_id x {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>} : i32
    // CHECK: ttatlas.fold_bias_into_dot = true
    %c = tt.get_program_id x {ttatlas.fold_bias_into_dot = true} : i32
    // The v2 slot: masked other (-inf spelled as a word), a vector, a transposed weight and a constant tile.
    // CHECK: #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20], other = -inf : bf16, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
    %f = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20], other = -inf : bf16, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>} : i32
    // CHECK: #ttatlas.slot<index = 2, kind = memory, arg = 3, role = bias_bf16, offset = 7, row_stride = 1, col_stride = 1, block = [0, 0, 1, 32], valid = [0, 0, 1, 32], other = 1.000000e+00 : bf16, vector_layout = cols, transposed = false, host_dtype = bf16, tile_dtype = bf16>
    %g = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 2, kind = memory, arg = 3, role = bias_bf16, offset = 7, row_stride = 1, col_stride = 1, block = [0, 0, 1, 32], valid = [0, 0, 1, 32], other = 1.0 : bf16, vector_layout = cols, transposed = false, host_dtype = bf16, tile_dtype = bf16>} : i32
    // CHECK: #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 64, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>
    %h = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = weight_fp8_nk, offset = 64, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32
    // CHECK: #ttatlas.slot<index = 3, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32640> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
    %i = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 3, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32640> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>} : i32
    return
  }
}
