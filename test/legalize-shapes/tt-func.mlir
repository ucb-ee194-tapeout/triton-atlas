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
