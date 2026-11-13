// RUN: not triton-atlas-opt %s --ttatlas-assign-slots 2>&1 | FileCheck %s
// CHECK: expected a static 32x32 tile or larger rank-two boundary tensor
module {
  tt.func @tile(%a: !tt.ptr<bf16>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %s = arith.constant dense<32> : tensor<16x1xi32>
    %scaled = arith.muli %rows, %s : tensor<16x1xi32>
    %rr = tt.broadcast %scaled : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %cols : tensor<1x16xi32> -> tensor<16x16xi32>
    %offsets = arith.addi %rr, %cc : tensor<16x16xi32>
    %base = tt.splat %a : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %p = tt.addptr %base, %offsets : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %x = tt.load %p {ttatlas.host_dtype = bf16} : tensor<16x16x!tt.ptr<bf16>>
    tt.return
  }
}
