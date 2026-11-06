// RUN: triton-atlas-opt %S/../fixtures/mlp.ttir.mlir --ttatlas-legalize-shapes | FileCheck %s

// Pass 3 is still a placeholder, so this TTIR fixture contains no linalg work
// for pass 4 to tile. Direct linalg coverage is in legalize-shapes.mlir.

// CHECK: tt.func public @mlp
// CHECK: tt.load
// CHECK: tt.trans
// CHECK: tt.dot
// CHECK: arith.maxnumf
// CHECK: tt.store
