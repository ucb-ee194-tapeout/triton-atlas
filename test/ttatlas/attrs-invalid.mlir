// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics

// The ttatlas verifier rejects each malformed tag where it is set.

// expected-error @below {{unknown tag ttatlas.foo}}
%a = tt.get_program_id x {ttatlas.foo = 1 : i32} : i32

// -----

// expected-error @below {{ttatlas.mxu must be an i32 in 0..1}}
%a = tt.get_program_id x {ttatlas.mxu = 2 : i32} : i32

// -----

// expected-error @below {{ttatlas.scale must be an i32 in 122..128}}
%a = tt.get_program_id x {ttatlas.scale = 130 : i32} : i32

// -----

// expected-error @below {{k_chain needs id >= 0, len >= 1 and 0 <= pos < len}}
%a = tt.get_program_id x {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 4, len = 4>} : i32

// -----

// expected-error @below {{tile_origin must have 2 results, no dims and 6 symbols}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<() -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{expected one of [activation_fp8, weight_fp8_nk, weight_fp8_kn, bias_bf16, mask_bf16, constant_bf16, output_bf16]}}
// expected-error @below {{failed to parse TTAtlas_SlotAttr parameter 'role'}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = weight, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{ttatlas.slot host_dtype disagrees with ttatlas.host_dtype}}
%a = tt.get_program_id x {ttatlas.host_dtype = bf16, ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : i32

// -----

// expected-error @below {{ttatlas.precision_policy must be a known policy name (fp8-bf16-v1)}}
module attributes {ttatlas.precision_policy = "fp32"} {}

// -----

// expected-error @below {{ttatlas.kernel must be a JSON object with "version": 1}}
func.func @k() attributes {ttatlas.kernel = "{\22version\22: 2}"} {
  return
}

// -----

// expected-error @below {{ttatlas.mxu must be an i32 in 0..1}}
%a = tt.get_program_id x {ttatlas.mxu = 0 : i64} : i32

// -----

// expected-error @below {{ttatlas.scale must be an i32 in 122..128}}
%a = tt.get_program_id x {ttatlas.scale = 127 : i8} : i32

// -----

// expected-error @below {{ttatlas.layout must be a #ttatlas<layout ...>}}
module attributes {ttatlas.layout = 1 : i32} {}

// -----

// expected-error @below {{ttatlas.dram_base must be a non-negative i64 address}}
module attributes {ttatlas.dram_base = -1 : i64} {}

// -----

// expected-error @below {{ttatlas.host_dtype must be a type}}
%a = tt.get_program_id x {ttatlas.host_dtype = 3 : i32} : i32

// -----

// expected-error @below {{k_chain needs id >= 0, len >= 1 and 0 <= pos < len}}
%a = tt.get_program_id x {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 0>} : i32

// -----

// expected-error @below {{logical_shape must be [rows, cols] with positive sizes}}
%a = tt.get_program_id x {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32]>} : i32

// -----

// expected-error @below {{slot host_dtype must be a scalar integer or float type}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = tensor<4xf32>>} : i32

// -----

// The agreement check works whatever the order of the two tags.
// expected-error @below {{ttatlas.slot host_dtype disagrees with ttatlas.host_dtype}}
%a = tt.get_program_id x {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>, ttatlas.host_dtype = bf16} : i32

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

// expected-error @below {{ttatlas.mxu must be an i32 in 0..1}}
func.func @res() -> (i32 {ttatlas.mxu = 7 : i32}) {
  %c = arith.constant 0 : i32
  return %c : i32
}

// -----

// expected-error @below {{ttatlas.legalized must be a list of {pass, what, loc} entries}}
module attributes {ttatlas.legalized = [{pass = "ttatlas-legalize-shapes", what = "mask -> padding"}]} {}

// -----

// Canonicalize rebuilds constants without their attributes, so tags on constants are rejected.
// expected-error @below {{ttatlas.mxu is on a constant; canonicalize drops attributes of constants, so tag the op that uses it}}
%a = arith.constant {ttatlas.mxu = 0 : i32} 0 : i32

// -----

// Only signless integers: a signed i32 is rejected, not read.
// expected-error @below {{ttatlas.mxu must be an i32 in 0..1}}
%a = tt.get_program_id x {ttatlas.mxu = 1 : si32} : i32
