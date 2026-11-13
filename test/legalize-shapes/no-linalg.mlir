// RUN: triton-atlas-opt %S/../inputs/mlp.ttir.mlir --ttatlas-legalize-shapes | FileCheck %s

// Pass 4 run on raw TTIR (pass 3 not run): there is no linalg op to tile, so
// the kernel comes out unchanged. Tiling coverage is in tiles.mlir.

// CHECK: tt.func public @mlp
// CHECK: tt.load
// CHECK: tt.trans
// CHECK: tt.dot
// CHECK: arith.maxnumf
// CHECK: tt.store
