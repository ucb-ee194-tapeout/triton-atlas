// RUN: not triton-atlas-opt %s --ttatlas-assign-slots 2>&1 | FileCheck %s
// CHECK: FP8 stores have no Atlas virtual output
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
    tt.store %p, %x {ttatlas.host_dtype = f8E4M3FN} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}
