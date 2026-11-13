// RUN: triton-atlas-opt %s --split-input-file --convert-triton-compute-to-linalg --ttatlas-verify-contract=stage=3 | FileCheck %s

// Elementwise group: arith / math on float tensors become linalg.generic through upstream convert-elementwise-to-linalg,
// each with a fresh tensor.empty output; integer address math stays TTIR. The stage-3 contract check runs after it.

// The ReLU of matmul + ReLU: maxnumf x, 0. Its output is a fresh tensor.empty, not the input (upstream reuses an
// operand, which made tile-and-fuse duplicate producers).
// CHECK-LABEL: tt.func @relu
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK-NEXT: %[[R:.*]] = linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[EMPTY]] : tensor<32x32xf32>)
// CHECK: arith.maxnumf
// CHECK: tt.return %[[R]]
tt.func @relu(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %r = arith.maxnumf %x, %zero : tensor<32x32xf32>
  tt.return %r : tensor<32x32xf32>
}

// -----

// Integer tensors are address math: they stay arith.
// CHECK-LABEL: tt.func @offsets
// CHECK-NOT: linalg.generic
// CHECK: arith.muli
// CHECK-NOT: linalg.generic
// CHECK: tt.return
tt.func @offsets(%r: tensor<32x32xi32>) -> tensor<32x32xi32> {
  %c32 = arith.constant dense<32> : tensor<32x32xi32>
  %o = arith.muli %r, %c32 : tensor<32x32xi32>
  tt.return %o : tensor<32x32xi32>
}

// -----

// tt.fp_to_fp and arith.truncf (BF16 <-> f32) become generics too.
// CHECK-LABEL: tt.func @conversions
// CHECK: linalg.generic
// CHECK: arith.extf %{{.*}} : bf16 to f32
// CHECK: linalg.generic
// CHECK: arith.truncf %{{.*}} : f32 to bf16
// CHECK-NOT: tt.fp_to_fp
tt.func @conversions(%x: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %e = tt.fp_to_fp %x : tensor<32x32xbf16> -> tensor<32x32xf32>
  %t = arith.truncf %e : tensor<32x32xf32> to tensor<32x32xbf16>
  tt.return %t : tensor<32x32xbf16>
}

// -----

// A widening tt.fp_to_fp with a rounding mode converts like any widening: it is exact, so the mode doesn't matter.
// CHECK-LABEL: tt.func @widen_rtz
// CHECK: linalg.generic
// CHECK: arith.extf %{{.*}} : bf16 to f32
// CHECK-NOT: tt.fp_to_fp
tt.func @widen_rtz(%x: tensor<32x32xbf16>) -> tensor<32x32xf32> {
  %e = tt.fp_to_fp %x, rounding = rtz : tensor<32x32xbf16> -> tensor<32x32xf32>
  tt.return %e : tensor<32x32xf32>
}

// -----

// ttatlas tags on an arith op end up on the generic, not on the scalar op in its body.
// CHECK-LABEL: tt.func @tags
// CHECK: linalg.generic {{.*}} attrs = {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 16]>}
// CHECK-NOT: ttatlas.pad
// CHECK: tt.return
tt.func @tags(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %r = arith.addf %x, %x {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 16]>} : tensor<32x32xf32>
  tt.return %r : tensor<32x32xf32>
}
