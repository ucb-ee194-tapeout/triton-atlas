// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s
// CHECK: role = bias_bf16
// CHECK-SAME: logical_shape = [1, 32], host_dtype = bf16
tt.func @bias(%a: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %s = tt.splat %a : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %p = tt.addptr %s, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %v = tt.load %p {ttatlas.host_dtype = bf16} : tensor<32x!tt.ptr<bf16>>
  tt.return
}
