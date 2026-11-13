// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --ttatlas-verify-contract=stage=5 | FileCheck %s

// Matmul chains, tags and arguments through pass 5, each followed by the stage-5 contract check.

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// A 2-matmul K chain with an f32 output: one readout after the last matmul only, and the output pointer becomes BF16.
// CHECK-LABEL: tt.func @k_chain_f32_out
// CHECK: %[[M0:.*]] = linalg.matmul {{.*}} -> tensor<32x32xf32>
// CHECK-NEXT: %[[M1:.*]] = linalg.matmul {{.*}} outs(%[[M0]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NOT: linalg.matmul
// CHECK: %[[R:.*]] = linalg.generic {{.*}} ins(%[[M1]] : tensor<32x32xf32>)
// CHECK: arith.truncf %{{.*}} : f32 to bf16
// CHECK: tt.store %{{.*}}, %[[R]] {ttatlas.host_dtype = f32} : tensor<32x32x!tt.ptr<bf16>>
tt.func @k_chain_f32_out(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %p: tensor<32x32x!tt.ptr<f32>>) {
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m0 = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m1 = linalg.matmul ins(%b, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%m0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.store %p, %m1 {ttatlas.host_dtype = f32} : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Two chains that share one fill (as CSE leaves them): each gets its own f32 fill and its own readout.
// CHECK-LABEL: tt.func @shared_fill
// CHECK: %[[F1:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x32xf32>)
// CHECK: %[[F2:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x32xf32>)
// CHECK: linalg.matmul {{.*}} outs(%[[F2]] : tensor<32x32xf32>)
// CHECK: linalg.matmul {{.*}} outs(%[[F1]] : tensor<32x32xf32>)
tt.func @shared_fill(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %p: tensor<32x32x!tt.ptr<bf16>>, %q: tensor<32x32x!tt.ptr<bf16>>) {
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m = linalg.matmul ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %n = linalg.matmul ins(%b, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %e1 = tensor.empty() : tensor<32x32xbf16>
  %ym = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%m : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.store %p, %ym : tensor<32x32x!tt.ptr<bf16>>
  %yn = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%n : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.store %q, %yn : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Pass 4 tags the stored value's defining op, here the final truncf generic, with ttatlas.pad. That generic becomes a
// pass-through and is erased; the tag moves to the op that now defines the stored value (the ReLU).
// CHECK-LABEL: tt.func @pad_moves
// CHECK: %[[RELU:.*]] = linalg.generic {{.*}} attrs = {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>}
// CHECK: arith.maxnumf %{{.*}}, %{{.*}} : bf16
// CHECK: tt.store %{{.*}}, %[[RELU]]
tt.func @pad_moves(%x: tensor<32x32xbf16>, %p: tensor<32x32x!tt.ptr<bf16>>) {
  %e0 = tensor.empty() : tensor<32x32xf32>
  %xf = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%e0 : tensor<32x32xf32>) {
  ^bb0(%in: bf16, %o: f32):
    %e = arith.extf %in : bf16 to f32
    linalg.yield %e : f32
  } -> tensor<32x32xf32>
  %z = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%xf, %z : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e0 : tensor<32x32xf32>) {
  ^bb0(%a: f32, %c: f32, %o: f32):
    %v = arith.maxnumf %a, %c : f32
    linalg.yield %v : f32
  } -> tensor<32x32xf32>
  %e1 = tensor.empty() : tensor<32x32xbf16>
  %y = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) attrs = {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>} {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.store %p, %y : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
