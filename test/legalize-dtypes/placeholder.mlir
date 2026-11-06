// RUN: triton-atlas-opt %S/../fixtures/mlp.ttir.mlir --ttatlas-legalize-dtypes | FileCheck %s

// Pass 4 (ttatlas-legalize-dtypes) is still a placeholder: it runs and leaves the kernel unchanged. This folder is where its
// tests go (one file per case); replace this file with the first real one.

// CHECK: tt.func public @mlp
// CHECK: tt.load
// CHECK: tt.trans
// CHECK: tt.dot
// CHECK: arith.maxnumf
// CHECK: tt.store
