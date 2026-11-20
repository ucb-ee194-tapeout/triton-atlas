// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots

// Pass 6 on kernels that have packs and unpacks (stage-3 form), rejecting what one slot array cannot describe.

// A computed tile stored to two pointers: its unpack carries one slot array, so the second store would be lost.
#map = affine_map<(d0, d1) -> (d0, d1)>
tt.func public @store_twice(%a: !tt.ptr<bf16>, %p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %rowoff = arith.muli %rows, %c32 : tensor<32x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rowb = tt.broadcast %rowoff : tensor<32x1xi32> -> tensor<32x32xi32>
  %colb = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
  %off = arith.addi %rowb, %colb : tensor<32x32xi32>
  %sa = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %pa = tt.addptr %sa, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  %x = tt.load %pa : tensor<32x32x!tt.ptr<bf16>>
  %e0 = tensor.empty() : tensor<32x32xf32>
  %xf = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xbf16>) outs(%e0 : tensor<32x32xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %v = arith.extf %in : bf16 to f32
    linalg.yield %v : f32
  } -> tensor<32x32xf32>
  %e1 = tensor.empty() : tensor<32x32xf32>
  // expected-error @below {{a stored tile written to two places}}
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%xf, %xf : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e1 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %in0: f32, %out: f32):
    %v = arith.addf %in, %in0 : f32
    linalg.yield %v : f32
  } -> tensor<32x32xf32>
  %e2 = tensor.empty() : tensor<32x32xbf16>
  %y = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%sum : tensor<32x32xf32>) outs(%e2 : tensor<32x32xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %v = arith.truncf %in : f32 to bf16
    linalg.yield %v : bf16
  } -> tensor<32x32xbf16>
  %sp = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %pp = tt.addptr %sp, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %pq = tt.addptr %sq, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  tt.store %pp, %y : tensor<32x32x!tt.ptr<bf16>>
  tt.store %pq, %y : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// One FP8 tile read as the activation and, transposed, as the weight of a matmul (x @ x^T) has no single role.
tt.func public @x_times_x(%a: !tt.ptr<f8E4M3FN>, %c: !tt.ptr<f32>) {
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %rowoff = arith.muli %rows, %c32 : tensor<32x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rowb = tt.broadcast %rowoff : tensor<32x1xi32> -> tensor<32x32xi32>
  %colb = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
  %off = arith.addi %rowb, %colb : tensor<32x32xi32>
  %sa = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %pa = tt.addptr %sa, %off : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  %x = tt.load %pa : tensor<32x32x!tt.ptr<f8E4M3FN>>
  %zero = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  // expected-error @below {{cannot infer a unique activation or weight role}}
  %m = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%x, %x : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %sc = tt.splat %c : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %pc = tt.addptr %sc, %off : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
  tt.store %pc, %m : tensor<32x32x!tt.ptr<f32>>
  tt.return
}
