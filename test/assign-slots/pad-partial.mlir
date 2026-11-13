// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s
// CHECK: logical_shape = [16, 32], host_dtype = bf16
module {
  tt.func @tile(%a: !tt.ptr<bf16>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %s = arith.constant dense<32> : tensor<32x1xi32>
    %scaled = arith.muli %rows, %s : tensor<32x1xi32>
    %rr = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x32xi32>
    %cc = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
    %offsets = arith.addi %rr, %cc : tensor<32x32xi32>
    %base = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %p = tt.addptr %base, %offsets : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %x = tt.load %p {ttatlas.host_dtype = bf16, ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
