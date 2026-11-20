// RUN: triton-atlas-opt %s --ttatlas-pack-blocks --canonicalize --cse | FileCheck %s

// Every op that pass 1 accepts on float tensors is either lane-local or a reader that step 4a protects (a row
// reduction over a padded column dim, and a contraction over a padded K). This kernel uses every accepted float
// tensor op on a padded 16x16 block: add, sub, mul, divf, the four min/max flavours, exp, exp2, extf, truncf, a
// transpose, a broadcast and the three row reductions. After packing, the only generics that mix lanes (that have a
// reduction iterator) are the three reductions, each reading a clamp, and no other op reads a padded lane into a
// valid one. If pass 1 starts to accept a new lane-mixing op, this test shows it reaching step 4b unprotected.
//
// CHECK-LABEL: func.func @all_ops
// The four clamp constants: two for the sum, one for the max, one for the min.
// CHECK-COUNT-4: arith.constant dense<"0x
// CHECK-COUNT-3: iterator_types = [{{.*}}"reduction"
// CHECK-NOT: iterator_types = [{{.*}}"reduction"
func.func @all_ops(%x: tensor<16x16xbf16>, %y: tensor<16x16xbf16>) -> (tensor<16xbf16>, tensor<16xbf16>, tensor<16xbf16>, tensor<16x16xbf16>) {
  %e = tensor.empty() : tensor<16x16xbf16>
  %t = linalg.transpose ins(%y : tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) permutation = [1, 0]
  %add = linalg.add ins(%x, %t : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %sub = linalg.sub ins(%add, %y : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %mul = linalg.mul ins(%sub, %x : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %div = linalg.div ins(%mul, %y : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %exp = linalg.exp ins(%div : tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) -> tensor<16x16xbf16>
  %mx = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%exp, %x : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%e : tensor<16x16xbf16>) {
  ^bb0(%a: bf16, %b: bf16, %o: bf16):
    %m = arith.maxnumf %a, %b : bf16
    %n = arith.minimumf %m, %b : bf16
    %p = arith.maximumf %n, %a : bf16
    %q = arith.minnumf %p, %b : bf16
    %f = arith.extf %q : bf16 to f32
    %g = math.exp2 %f : f32
    %h = arith.truncf %g : f32 to bf16
    linalg.yield %h : bf16
  } -> tensor<16x16xbf16>
  %zero = arith.constant 0.0 : bf16
  %ninf = arith.constant 0xFF80 : bf16
  %pinf = arith.constant 0x7F80 : bf16
  %v = tensor.empty() : tensor<16xbf16>
  %z0 = linalg.fill ins(%zero : bf16) outs(%v : tensor<16xbf16>) -> tensor<16xbf16>
  %z1 = linalg.fill ins(%ninf : bf16) outs(%v : tensor<16xbf16>) -> tensor<16xbf16>
  %z2 = linalg.fill ins(%pinf : bf16) outs(%v : tensor<16xbf16>) -> tensor<16xbf16>
  %sum = linalg.reduce { arith.addf } ins(%mx : tensor<16x16xbf16>) outs(%z0 : tensor<16xbf16>) dimensions = [1]
  %max = linalg.reduce { arith.maxnumf } ins(%mx : tensor<16x16xbf16>) outs(%z1 : tensor<16xbf16>) dimensions = [1]
  %min = linalg.reduce { arith.minnumf } ins(%mx : tensor<16x16xbf16>) outs(%z2 : tensor<16xbf16>) dimensions = [1]
  %b = linalg.broadcast ins(%sum : tensor<16xbf16>) outs(%e : tensor<16x16xbf16>) dimensions = [1]
  return %sum, %max, %min, %b : tensor<16xbf16>, tensor<16xbf16>, tensor<16xbf16>, tensor<16x16xbf16>
}
