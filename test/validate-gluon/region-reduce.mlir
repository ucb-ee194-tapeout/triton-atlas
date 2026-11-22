// RUN: triton-atlas-opt %s --ttatlas-validate-gluon --ttatlas-lower-gluon-layouts-to-ttir | FileCheck %s
//
// Layout-only normalization must preserve a real region and its terminator.
// The prior conversion implementation rejected this tt.reduce outright.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module {
  tt.func @row_sum(%x: tensor<16x16xf32, #blocked>) -> tensor<16xf32, #blocked> {
    %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #blocked>
    tt.return %r : tensor<16xf32, #blocked>
  }
}

// CHECK: tt.func @row_sum(%[[X:.*]]: tensor<16x16xf32>) -> tensor<16xf32>
// CHECK: "tt.reduce"(%[[X]])
// CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK: tt.reduce.return
// CHECK-NOT: #ttg.blocked
