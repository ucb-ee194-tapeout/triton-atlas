// RUN: triton-atlas-opt %S/../fixtures/boundary-slices.ttir.mlir --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 | FileCheck %s
// CHECK-DAG: affine_map<()[s0, s1, s2, s3, s4, s5] -> (1, 0)>
// CHECK-DAG: affine_map<()[s0, s1, s2, s3, s4, s5] -> (1, 32)>
// CHECK: tt.load
// CHECK-NOT: ttatlas.slot
// CHECK: tensor.extract_slice
// CHECK-SAME: index = 0, arg = 0, role = constant_bf16
// CHECK: tensor.extract_slice
// CHECK-SAME: index = 1, arg = 0, role = constant_bf16
// CHECK: tensor.extract_slice
// CHECK-SAME: index = 0, arg = 0, role = constant_bf16
// CHECK: tensor.insert_slice
// CHECK-SAME: index = 0, arg = 1, role = output_bf16
// CHECK-SAME: host_dtype = f32
// CHECK: tensor.insert_slice
// CHECK-SAME: index = 1, arg = 1, role = output_bf16
// CHECK: tt.store
// CHECK-NOT: ttatlas.slot
