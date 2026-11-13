// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-shapes --canonicalize --cse --ttatlas-verify-contract=stage=4 | FileCheck %s

// Tiling and padding cases legalize-shapes.mlir doesn't cover, each checked against the stage-4 contract too. The ops
// are in the forms pass 3 emits (elementwise and conversions as linalg.generic), in func.funcs so a tiling failure in
// a tt.func (tt-func.mlir) doesn't hide these.

// A K of 96: one output tile, a K chain of 3.
// CHECK-LABEL: func.func @split_k_96
// CHECK-COUNT-3: linalg.matmul {{.*}}ins(%{{.*}}, %{{.*}} : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
// CHECK-NOT: linalg.matmul
func.func @split_k_96(%a: tensor<32x96xf8E4M3FN>, %b: tensor<96x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %r = linalg.matmul ins(%a, %b : tensor<32x96xf8E4M3FN>, tensor<96x32xf8E4M3FN>)
                     outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}

// -----

// A K of 40, not a multiple of 32: both operands are padded with zero along K, then a K chain of 2.
// CHECK-LABEL: func.func @split_k_40
// CHECK: ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 40]>
// CHECK: ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [40, 32]>
// CHECK-COUNT-2: linalg.matmul {{.*}}ins(%{{.*}}, %{{.*}} : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
// CHECK-NOT: linalg.matmul
func.func @split_k_40(%a: tensor<32x40xf8E4M3FN>, %b: tensor<40x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %r = linalg.matmul ins(%a, %b : tensor<32x40xf8E4M3FN>, tensor<40x32xf8E4M3FN>)
                     outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}

// -----

// A max over 50 columns, the padded dimension being the reduced one: the padded lanes hold -inf.
// CHECK-LABEL: func.func @max_over_padded_columns
// CHECK: ttatlas.pad = #ttatlas.pad<neutral = neg_inf, logical_shape = [32, 50]>
// CHECK-COUNT-2: linalg.reduce
func.func @max_over_padded_columns(%x: tensor<32x50xf32>) -> tensor<32xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
  %r = linalg.reduce ins(%x : tensor<32x50xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  return %r : tensor<32xf32>
}

// -----

// A min over 50 columns needs +inf in the padded lanes: with 0, a padded lane wins whenever every real value is
// positive.
// CHECK-LABEL: func.func @min_over_padded_columns
// CHECK: ttatlas.pad = #ttatlas.pad<neutral = pos_inf, logical_shape = [32, 50]>
// CHECK-COUNT-2: linalg.reduce
func.func @min_over_padded_columns(%x: tensor<32x50xf32>) -> tensor<32xf32> {
  %pinf = arith.constant 0x7F800000 : f32
  %empty = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%pinf : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
  %r = linalg.reduce ins(%x : tensor<32x50xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.minnumf %a, %b : f32
    linalg.yield %m : f32
  }
  return %r : tensor<32xf32>
}

// -----

// A conversion (BF16 -> f32, as every BF16 kernel has) that needs padding: an elementwise op never reads its padded
// lanes back, so any neutral will do; pass 4 must pad it, not reject it.
#id = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @pad_conversion
// CHECK: arith.extf
// CHECK-NOT: arith.extf
func.func @pad_conversion(%x: tensor<16x16xbf16>) -> tensor<16x16xf32> {
  %empty = tensor.empty() : tensor<16x16xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<16x16xbf16>) outs(%empty : tensor<16x16xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %e = arith.extf %in : bf16 to f32
    linalg.yield %e : f32
  } -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}

// -----

// A 64x64x64 matmul feeding a ReLU and a truncf, as in matmul + ReLU kernels: the matmul's K must still be split when
// it is fused into its consumer's tiles, so every matmul is one 32x32x32 tile (2x2 output tiles, K chains of 2).
#id2 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_relu_truncf_64
// CHECK-COUNT-8: linalg.matmul {{.*}}ins(%{{.*}}, %{{.*}} : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
// CHECK-NOT: linalg.matmul
func.func @matmul_relu_truncf_64(%a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>) -> tensor<64x64xbf16> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>)
                      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %e1 = tensor.empty() : tensor<64x64xf32>
  %relu = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%e1 : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.maxnumf %in, %zero : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  %e2 = tensor.empty() : tensor<64x64xbf16>
  %t = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%relu : tensor<64x64xf32>) outs(%e2 : tensor<64x64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %c = arith.truncf %in : f32 to bf16
    linalg.yield %c : bf16
  } -> tensor<64x64xbf16>
  return %t : tensor<64x64xbf16>
}

// -----

// A 16x16 load and store, smaller than a tile (what add_16 in test/from-jit is): the load is read through one tensor.pad
// up to a tile, which carries the tile's real shape, and the stored value is a 16x16 crop of the tile whose producer
// carries the same tag. The stage-4 contract accepts both ends.
// CHECK-LABEL: tt.func @sub_tile_add
// CHECK: tt.load {{.*}} : tensor<16x16x!tt.ptr<bf16>>
// CHECK: tensor.pad
// CHECK: ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 16]>
// CHECK: linalg.add {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 16]>}
// CHECK: tensor.extract_slice {{.*}} [16, 16] [1, 1]
// CHECK: tt.store
tt.func @sub_tile_add(%in: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
  %ins = tt.splat %in : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
  %x = tt.load %ins : tensor<16x16x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<16x16xbf16>
  %y = linalg.add ins(%x, %x : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %outs = tt.splat %out : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
  tt.store %outs, %y : tensor<16x16x!tt.ptr<bf16>>
  tt.return
}

// -----

// A row (a bias) of 16: the tag is still rank 2, [1, 16], never [16].
// CHECK-LABEL: tt.func @bias_row_16
// CHECK: tensor.pad
// CHECK: ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [1, 16]>
// CHECK: linalg.add {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [1, 16]>}
// CHECK-NOT: logical_shape = [16]
tt.func @bias_row_16(%in: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
  %ins = tt.splat %in : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>>
  %x = tt.load %ins : tensor<16x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<16xbf16>
  %y = linalg.add ins(%x, %x : tensor<16xbf16>, tensor<16xbf16>) outs(%e : tensor<16xbf16>) -> tensor<16xbf16>
  %outs = tt.splat %out : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>>
  tt.store %outs, %y : tensor<16x!tt.ptr<bf16>>
  tt.return
}

// -----

// Padded lanes that an op changed: a 32x40 value through exp, then a row sum over the 40 columns. The host fills only a
// kernel input, so the sum cannot count on the zeros that padding wrote after exp (exp(0) is 1 on the chip): it reads
// the padded value times a constant 1/0 mask. Tiling cuts the constant into one tile per column tile.
#id = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @exp_then_row_sum_40
// CHECK-DAG: %[[REAL:.*]] = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-DAG: %[[EDGE:.*]] = arith.constant dense<"{{.*}}"> : tensor<32x32xf32>
// CHECK: math.exp
// CHECK: linalg.mul ins(%{{.*}}, %[[REAL]] : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK: linalg.mul ins(%{{.*}}, %[[EDGE]] : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK-NOT: linalg.mul
func.func @exp_then_row_sum_40(%x: tensor<32x40xf32>) -> tensor<32xf32> {
  %e = tensor.empty() : tensor<32x40xf32>
  %exp = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<32x40xf32>) outs(%e : tensor<32x40xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.exp %in : f32
    linalg.yield %v : f32
  } -> tensor<32x40xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
  %r = linalg.reduce ins(%exp : tensor<32x40xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  }
  return %r : tensor<32xf32>
}

// -----

// A row max straight from the input needs no mask: the host fills the padded columns with -inf (the tag), and nothing
// between the input and the max changes them.
// CHECK-LABEL: func.func @row_max_40_from_input
// CHECK-NOT: linalg.mul
// CHECK-NOT: linalg.add
func.func @row_max_40_from_input(%x: tensor<32x40xf32>) -> tensor<32xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
  %r = linalg.reduce ins(%x : tensor<32x40xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  return %r : tensor<32xf32>
}
