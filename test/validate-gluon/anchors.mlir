// RUN: triton-atlas-opt %s --ttatlas-validate-gluon --ttatlas-lower-gluon-layouts-to-ttir | FileCheck %s
//
// This is deliberately real GLIR syntax emitted by Gluon's frontend: an
// AutoLayout range becomes concrete only at the set_auto_layout anchor.

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "atlas:virtual", "ttg.threads-per-warp" = 1 : i32} {
  tt.func @anchor() {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #gluon.auto_encoding>
    %1 = gluon.set_auto_layout %0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    tt.return
  }
}

// CHECK-NOT: gluon.set_auto_layout
// CHECK-NOT: #gluon.auto_encoding
// CHECK-NOT: #ttg.blocked
// CHECK: tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
