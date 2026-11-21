// RUN: not triton-atlas-opt %s --ttatlas-validate-gluon 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16, 1, 1], threadsPerWarp = [1, 1, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "atlas:virtual", "ttg.threads-per-warp" = 1 : i32} {
  tt.func @rank3() {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #gluon.auto_encoding>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #gluon.auto_encoding> -> tensor<16x1xi32, #gluon.auto_encoding>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<16x1xi32, #gluon.auto_encoding> -> tensor<16x1x1xi32, #gluon.auto_encoding>
    %3 = gluon.set_auto_layout %2 : tensor<16x1x1xi32, #gluon.auto_encoding> -> tensor<16x1x1xi32, #blocked>
    tt.return
  }
}

// CHECK: error: atlas: unsupported gluon.set_auto_layout: Gluon set_auto_layout has a tensor rank other than 1 or 2
