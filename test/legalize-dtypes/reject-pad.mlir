// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-dtypes --verify-diagnostics

// ttatlas.pad on an erased pass-through moves to the value's producer (see chains.mlir); pass 5 fails rather than drop
// or spread it.

// The producer has another user, which would then look padded too.
#map = affine_map<(d0, d1) -> (d0, d1)>

tt.func @pad_spreads(%x: tensor<32x32xbf16>, %p: tensor<32x32x!tt.ptr<bf16>>, %q: tensor<32x32x!tt.ptr<bf16>>) {
  %e0 = tensor.empty() : tensor<32x32xf32>
  %e1 = tensor.empty() : tensor<32x32xbf16>
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
  // expected-error @below {{atlas: unsupported linalg.generic: ttatlas.pad would also apply to the other users of the value that replaces this op}}
  %y = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) attrs = {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>} {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  %y2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.store %p, %y : tensor<32x32x!tt.ptr<bf16>>
  tt.store %q, %y2 : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// The producer already has a different pad.
#map = affine_map<(d0, d1) -> (d0, d1)>

tt.func @pad_conflict(%x: tensor<32x32xbf16>, %p: tensor<32x32x!tt.ptr<bf16>>, %q: tensor<32x32x!tt.ptr<bf16>>) {
  %e0 = tensor.empty() : tensor<32x32xf32>
  %e1 = tensor.empty() : tensor<32x32xbf16>
  %xf = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%e0 : tensor<32x32xf32>) {
  ^bb0(%in: bf16, %o: f32):
    %e = arith.extf %in : bf16 to f32
    linalg.yield %e : f32
  } -> tensor<32x32xf32>
  %z = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%xf, %z : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e0 : tensor<32x32xf32>) attrs = {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 16]>} {
  ^bb0(%a: f32, %c: f32, %o: f32):
    %v = arith.maxnumf %a, %c : f32
    linalg.yield %v : f32
  } -> tensor<32x32xf32>
  // expected-error @below {{atlas: unsupported linalg.generic: ttatlas.pad differs from the one on the value that replaces this op}}
  %y = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) attrs = {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>} {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x32xbf16>
  tt.store %p, %y : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
