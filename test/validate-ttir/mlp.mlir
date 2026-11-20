// RUN: triton-atlas-opt %S/../inputs/mlp.ttir.mlir --ttatlas-validate-ttir --ttatlas-verify-contract=stage=1 | FileCheck %s

// Pass 1 accepts the 32x32 MLP and tags every load and store with the host tensor's dtype (x and w are FP8, the bias
// and the output BF16). The stage-1 contract check runs after it.

// CHECK: tt.func public @mlp
// CHECK: tt.load %{{.*}}
// CHECK: tt.load %{{.*}}
// CHECK: tt.load %{{.*}}
// CHECK: tt.trans
// CHECK: tt.dot
// CHECK: arith.maxnumf
// CHECK: tt.store %{{.*}}, %{{.*}}
