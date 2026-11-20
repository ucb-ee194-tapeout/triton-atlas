// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics

// The ttatlas verifier rejects each malformed tag where it is set.

// expected-error @below {{unknown tag ttatlas.foo}}
%a = tt.get_program_id x {ttatlas.foo = 1 : i32} : i32

// -----

// expected-error @below {{k_chain needs id >= 0, len >= 1 and 0 <= pos < len}}
%a = tt.get_program_id x {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 4, len = 4>} : i32

// -----

// expected-error @below {{slot valid box must lie inside its block}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 16, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{slot block must be a non-empty and valid a possibly empty}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 33, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{unknown role}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = weight, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{unknown vector_layout}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = diag, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{transposed must be true or false}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = maybe, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{slot host_dtype and tile_dtype must be scalar}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = tensor<4xf32>, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{a constant slot needs a pattern of type tensor<32x32xi16>}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<0> : tensor<32x32xi8>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>} : i32

// -----

// expected-error @below {{slot other must have the host or tile dtype}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], other = 1.0 : bf16, vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{ttatlas.precision_policy must be a known policy name (fp8-bf16-v1)}}
module attributes {ttatlas.precision_policy = "fp32"} {}

// -----

// expected-error @below {{ttatlas.kernel must be a JSON object with "version": 2}}
func.func @k() attributes {ttatlas.kernel = "{\22version\22: 1}"} {
  return
}

// -----

// expected-error @below {{ttatlas.layout must be a #ttatlas<layout ...>}}
module attributes {ttatlas.layout = 1 : i32} {}

// -----

// expected-error @below {{ttatlas.dram_base must be a non-negative i64 address}}
module attributes {ttatlas.dram_base = -1 : i64} {}

// -----

// expected-error @below {{k_chain needs id >= 0, len >= 1 and 0 <= pos < len}}
%a = tt.get_program_id x {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 0>} : i32

// -----

// expected-error @below {{ttatlas.kernel is not valid JSON}}
func.func @k() attributes {ttatlas.kernel = "{version"} {
  return
}

// -----

// Tags on function arguments and results are checked too.
// expected-error @below {{unknown tag ttatlas.bad}}
func.func @arg(%x: i32 {ttatlas.bad = 1 : i32}) {
  return
}

// -----

// expected-error @below {{ttatlas.fold_bias_into_dot must be a bool}}
func.func @res() -> (i32 {ttatlas.fold_bias_into_dot = 7 : i32}) {
  %c = arith.constant 0 : i32
  return %c : i32
}

// -----

// Canonicalize rebuilds constants without their attributes, so tags on constants are rejected.
// expected-error @below {{ttatlas.fold_bias_into_dot is on a constant; canonicalize drops attributes of constants, so tag the op that uses it}}
%a = arith.constant {ttatlas.fold_bias_into_dot = true} 0 : i32
