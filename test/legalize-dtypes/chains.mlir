// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --ttatlas-verify-contract=stage=5 | FileCheck %s

// Matmul chains, tags and arguments through pass 5, each followed by the stage-5 contract check.

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
