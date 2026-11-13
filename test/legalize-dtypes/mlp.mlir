// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --ttatlas-verify-contract=stage=5 | FileCheck %s

// The MLP fixture (relu(x @ w.T + b), one 32x32 tile) as pass 4 leaves it, hand-written: compute in linalg, f32 after
// the dot. Pass 5 keeps the matmul's accumulator f32 (it stays in the MXU), adds the readout marker after it, retypes
// the bias add and the ReLU to BF16, and erases the bias extf and the final truncf, which became BF16 -> BF16. The
// stage-5 contract check runs after it.

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0) -> (d0)>

// CHECK-LABEL: tt.func public @mlp
// The accumulator's zero, tensor.empty and fill stay f32.
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[E0:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[F:.*]] = linalg.fill ins(%[[Z]] : f32) outs(%[[E0]] : tensor<32x32xf32>)
// CHECK: %[[M:.*]] = linalg.matmul {{.*}} outs(%[[F]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT: %[[E:.*]] = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT: %[[READ:.*]] = linalg.generic {{.*}} ins(%[[M]] : tensor<32x32xf32>) outs(%[[E]] : tensor<32x32xbf16>)
// CHECK: arith.truncf %{{.*}} : f32 to bf16
// CHECK: %[[BIAS:.*]] = linalg.broadcast ins(%[[B:.*]] : tensor<32xbf16>) outs(%{{.*}} : tensor<32x32xbf16>)
// CHECK: %[[SUM:.*]] = linalg.generic {{.*}} ins(%[[READ]], %[[BIAS]] : tensor<32x32xbf16>, tensor<32x32xbf16>)
// CHECK: arith.addf %{{.*}}, %{{.*}} : bf16
// CHECK: %[[RELU:.*]] = linalg.generic {{.*}} ins(%[[SUM]], %{{.*}} : tensor<32x32xbf16>, tensor<32x32xbf16>)
// CHECK: arith.maxnumf %{{.*}}, %{{.*}} : bf16
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: tt.store %{{.*}}, %[[RELU]] {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
tt.func public @mlp(%xp: !tt.ptr<f8E4M3FN>, %wp: !tt.ptr<f8E4M3FN>, %bp: !tt.ptr<bf16>, %yp: !tt.ptr<bf16>) {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %rows = arith.muli %r1, %c32 : tensor<32x1xi32>
  %r0 = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rb = tt.broadcast %rows : tensor<32x1xi32> -> tensor<32x32xi32>
  %cb = tt.broadcast %r0 : tensor<1x32xi32> -> tensor<32x32xi32>
  %off = arith.addi %rb, %cb : tensor<32x32xi32>
  %xs = tt.splat %xp : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %xa = tt.addptr %xs, %off : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  %x = tt.load %xa {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
  %ws = tt.splat %wp : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %wa = tt.addptr %ws, %off : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  %w = tt.load %wa {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
  %bs = tt.splat %bp : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %ba = tt.addptr %bs, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %b = tt.load %ba {ttatlas.host_dtype = bf16} : tensor<32x!tt.ptr<bf16>>
  %eb = tensor.empty() : tensor<32xf32>
  %bf = linalg.generic {indexing_maps = [#row, #row], iterator_types = ["parallel"]} ins(%b : tensor<32xbf16>) outs(%eb : tensor<32xf32>) {
  ^bb0(%in: bf16, %o: f32):
    %e = arith.extf %in : bf16 to f32
    linalg.yield %e : f32
  } -> tensor<32xf32>
  %et = tensor.empty() : tensor<32x32xf8E4M3FN>
  %wt = linalg.transpose ins(%w : tensor<32x32xf8E4M3FN>) outs(%et : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%x, %wt : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %bb = linalg.broadcast ins(%bf : tensor<32xf32>) outs(%e0 : tensor<32x32xf32>) dimensions = [0]
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%m, %bb : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e0 : tensor<32x32xf32>) {
  ^bb0(%a: f32, %c: f32, %o: f32):
    %s = arith.addf %a, %c : f32
    linalg.yield %s : f32
  } -> tensor<32x32xf32>
  %z = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%sum, %z : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e0 : tensor<32x32xf32>) {
  ^bb0(%a: f32, %c: f32, %o: f32):
    %v = arith.maxnumf %a, %c : f32
    linalg.yield %v : f32
  } -> tensor<32x32xf32>
  %e1 = tensor.empty() : tensor<32x32xbf16>
  %y = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  %ys = tt.splat %yp : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %ya = tt.addptr %ys, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  tt.store %ya, %y {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// An f32 output: Atlas writes BF16, so the output pointer becomes BF16 too and tt.store stays valid; the host dtype
// stays f32, so the driver converts back.
// CHECK-LABEL: tt.func public @f32_output
// CHECK-SAME: (%{{.*}}: !tt.ptr<f8E4M3FN>, %[[C:.*]]: !tt.ptr<bf16>)
// CHECK: %[[READ:.*]] = linalg.generic {{.*}} outs(%{{.*}} : tensor<32x32xbf16>)
// CHECK: tt.splat %[[C]] : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
// CHECK: tt.store %{{.*}}, %[[READ]] {ttatlas.host_dtype = f32} : tensor<32x32x!tt.ptr<bf16>>
tt.func public @f32_output(%ap: !tt.ptr<f8E4M3FN>, %cp: !tt.ptr<f32>) {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %r0 = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %off = tt.broadcast %r0 : tensor<1x32xi32> -> tensor<32x32xi32>
  %as = tt.splat %ap : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %aa = tt.addptr %as, %off : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  %a = tt.load %aa {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m = linalg.matmul ins(%a, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %cs = tt.splat %cp : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %ca = tt.addptr %cs, %off : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
  tt.store %ca, %m {ttatlas.host_dtype = f32} : tensor<32x32x!tt.ptr<f32>>
  tt.return
}
