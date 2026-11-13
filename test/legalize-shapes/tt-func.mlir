// RUN: triton-atlas-opt %s --split-input-file --ttatlas-legalize-shapes --canonicalize --cse --ttatlas-verify-contract=stage=4 | FileCheck %s

// Pass 4 tiling inside a tt.func, the function pass 3 hands it (the other tests here use func.func). Each output must
// also pass the stage-4 contract check.

// A 64x64x64 matmul inside a tt.func: 2x2 output tiles, each a K chain of 2 matmuls.
// CHECK-LABEL: tt.func @matmul_64_in_tt_func
// CHECK-COUNT-8: linalg.matmul {{.*}}ttatlas.mxu = 0 : i32
// CHECK-NOT: linalg.matmul
// CHECK: tt.return
tt.func @matmul_64_in_tt_func(%a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.matmul ins(%a, %b : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>)
                     outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  tt.return %r : tensor<64x64xf32>
}

// -----

// A row max over 64 columns inside a tt.func: 2 row tiles, each reducing 2 column tiles.
// CHECK-LABEL: tt.func @row_max_64_in_tt_func
// CHECK-COUNT-4: linalg.reduce
// CHECK: tt.return
tt.func @row_max_64_in_tt_func(%x: tensor<64x64xf32>) -> tensor<64xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<64xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  %r = linalg.reduce ins(%x : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  tt.return %r : tensor<64xf32>
}

// -----

// One 16x16 load read by a row max and by the subtraction x - max (the start of a softmax): the max wants -inf in the
// padded lanes and the subtraction does not care, but the host fills a load once, so both pads of the load carry the
// same tag (zero, which is finite) and the max reads an add mask. Two different tags on one load would not pass the
// stage-4 check.
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: tt.func @one_fill_for_max_and_subtract
// CHECK: tt.load
// CHECK: } {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 16]>} : tensor<16x16xf32> to tensor<32x32xf32>
// CHECK-NOT: } {ttatlas.pad = #ttatlas.pad<neutral = neg_inf, logical_shape = [16, 16]>}
// CHECK: linalg.add {ttatlas.mask_apply}
tt.func @one_fill_for_max_and_subtract(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  %x = tt.load %ptrs : tensor<16x16x!tt.ptr<f32>>
  %ninf = arith.constant 0xFF800000 : f32
  %e1 = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
  %max = linalg.reduce ins(%x : tensor<16x16xf32>) outs(%init : tensor<16xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  %e = tensor.empty() : tensor<16x16xf32>
  %shifted = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x, %max : tensor<16x16xf32>, tensor<16xf32>) outs(%e : tensor<16x16xf32>) {
  ^bb0(%in: f32, %m: f32, %o: f32):
    %d = arith.subf %in, %m : f32
    linalg.yield %d : f32
  } -> tensor<16x16xf32>
  %optrs = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  tt.store %optrs, %shifted : tensor<16x16x!tt.ptr<f32>>
  tt.return
}
