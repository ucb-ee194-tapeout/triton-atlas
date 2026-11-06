// RUN: triton-atlas-opt %s --ttatlas-assign-slots --ttatlas-verify-contract=stage=5 | FileCheck %s
// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s
// CHECK: ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = #map, logical_shape = [32, 32], host_dtype = f8E4M3FN>
module {
  tt.func @tile(%a: !tt.ptr<f8E4M3FN>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %s = arith.constant dense<32> : tensor<32x1xi32>
    %scaled = arith.muli %rows, %s : tensor<32x1xi32>
    %rr = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x32xi32>
    %cc = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
    %offsets = arith.addi %rr, %cc : tensor<32x32xi32>
    %base = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %p = tt.addptr %base, %offsets : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %x = tt.load %p {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %zero = arith.constant dense<0.0> : tensor<32x32xf8E4M3FN>
    %acc = arith.constant dense<0.0> : tensor<32x32xf32>
    %dot = tt.dot %x, %zero, %acc : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    tt.return
  }
}
