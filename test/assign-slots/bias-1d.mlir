// RUN: not triton-atlas-opt %s --ttatlas-assign-slots 2>&1 | FileCheck %s
// CHECK: 1-D bias needs the Stage B driver-broadcast contract
tt.func @bias(%a: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %s = tt.splat %a : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %p = tt.addptr %s, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %v = tt.load %p {ttatlas.host_dtype = bf16} : tensor<32x!tt.ptr<bf16>>
  tt.return
}
