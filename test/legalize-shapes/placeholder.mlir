// RUN: triton-atlas-opt %S/../fixtures/mlp.ttir.mlir --ttatlas-legalize-shapes | FileCheck %s

// Pass 3 (ttatlas-legalize-shapes) is still a placeholder: it runs and leaves the kernel unchanged. This folder is where its
// tests go (one file per case); replace this file with the first real one.

// CHECK: tt.func public @mlp
// CHECK: tt.load
// CHECK: tt.trans
// CHECK: tt.dot
// CHECK: arith.maxnumf
// CHECK: tt.store
