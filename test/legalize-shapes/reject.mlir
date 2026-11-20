// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-pack-blocks

// Step 4a rejects what its rules do not cover, with a name, and never packs it wrongly.

// A contraction whose K is padded needs the host's zero in the padded K lanes, which only a load operand has.
func.func @padded_k_from_compute(%a: tensor<32x16xf8E4M3FN>, %b: tensor<16x32xf8E4M3FN>, %c: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // expected-error @below {{a contraction with a padded K must read both operands straight from loads}}
  %r = linalg.matmul ins(%a, %b : tensor<32x16xf8E4M3FN>, tensor<16x32xf8E4M3FN>) outs(%c : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %r : tensor<32x32xbf16>
}

// -----

// A padded row product has no raw-bit clamp rule.
func.func @padded_product(%x: tensor<16x16xbf16>, %init: tensor<16xbf16>) -> tensor<16xbf16> {
  // expected-error @below {{a padded reduction needs a single BF16 add, max or min combiner over one input}}
  %r = linalg.reduce { arith.mulf } ins(%x : tensor<16x16xbf16>) outs(%init : tensor<16xbf16>) dimensions = [1]
  return %r : tensor<16xbf16>
}

// -----

// A padded row sum in f32 would need its own clamp constants; pass 5 makes every compute value BF16 first.
func.func @padded_f32_sum(%x: tensor<16x16xf32>, %init: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error @below {{a padded reduction needs a single BF16 add, max or min combiner over one input}}
  %r = linalg.reduce { arith.addf } ins(%x : tensor<16x16xf32>) outs(%init : tensor<16xf32>) dimensions = [1]
  return %r : tensor<16xf32>
}

// -----

// A constant that is not a splat cannot be packed at compile time.
func.func @packed_constant(%x: tensor<16x16xbf16>) -> tensor<16x16xbf16> {
  %c = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xbf16>
  %e = tensor.empty() : tensor<2x2xbf16>
  // expected-error @below {{a pack of a constant that is not a splat}}
  %r = linalg.add ins(%c, %c : tensor<2x2xbf16>, tensor<2x2xbf16>) outs(%e : tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %x : tensor<16x16xbf16>
}
