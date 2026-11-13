// RUN: triton-atlas-opt %s --split-input-file --convert-triton-compute-to-linalg --verify-diagnostics

// Float compute pass 3 has no linalg form for fails with a reason, instead of being wrapped unconverted in a
// linalg.generic or left behind. Pass 1 rejects all of these first; pass 3 keeps the rule when it runs alone.

// A narrowing tt.fp_to_fp that doesn't round to nearest even (arith.truncf's rounding).
tt.func @fp_to_fp_rtz(%x: tensor<32x32xf32>) -> tensor<32x32xbf16> {
  // expected-error @below {{atlas: unsupported tt.fp_to_fp: converts with no arith form (equal widths, or narrowing that isn't rtne)}}
  %r = tt.fp_to_fp %x, rounding = rtz : tensor<32x32xf32> -> tensor<32x32xbf16>
  tt.return %r : tensor<32x32xbf16>
}

// -----

// A float tt op with no pattern here.
tt.func @precise_sqrt(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{atlas: unsupported tt.precise_sqrt: pass 3 has no linalg form for this float op}}
  %r = tt.precise_sqrt %x : tensor<32x32xf32>
  tt.return %r : tensor<32x32xf32>
}

// -----

// A tt.fp_to_fp between equal widths (BF16 to f16): no arith cast does that.
tt.func @fp_to_fp_equal_width(%x: tensor<32x32xbf16>) -> tensor<32x32xf16> {
  // expected-error @below {{atlas: unsupported tt.fp_to_fp: converts with no arith form (equal widths, or narrowing that isn't rtne)}}
  %r = tt.fp_to_fp %x, rounding = rtne : tensor<32x32xbf16> -> tensor<32x32xf16>
  tt.return %r : tensor<32x32xf16>
}

// -----

// An external library call on floats.
tt.func @extern(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{atlas: unsupported tt.extern_elementwise: pass 3 has no linalg form for this float op}}
  %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_expf"} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return %r : tensor<32x32xf32>
}

// -----

// A float reduce down to a scalar: pass 3 converts row reductions of a tile only.
tt.func @full_reduce(%x: tensor<32xf32>) -> f32 {
  // expected-error @below {{atlas: unsupported tt.reduce: reduces to a scalar; pass 3 converts row reductions of a tile only}}
  %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<32xf32>) -> f32
  tt.return %r : f32
}
