// RUN: triton-atlas-opt %S/../fixtures/mm32.ttir.mlir --ttatlas-normalize --ttatlas-verify-contract=stage=2 | FileCheck %s

// mm32's K loop (4 trips, K = 128) is unrolled into one chain of 4 dots, each accumulating onto the previous one and
// tagged with its place in the chain. The stage-2 contract check runs after it.

// CHECK-LABEL: tt.func public @mm32
// CHECK-NOT: scf.for
// CHECK: %[[D0:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 4>}
// CHECK: %[[D1:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D0]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 4>}
// CHECK: %[[D2:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D1]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 2, len = 4>}
// CHECK: %[[D3:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D2]] {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 3, len = 4>}
// CHECK-NOT: scf.for
// CHECK: arith.truncf %[[D3]]
