// RUN: triton-atlas-opt %s --split-input-file --ttatlas-normalize | FileCheck %s --check-prefixes=CHECK,FLAG-OFF
// RUN: triton-atlas-opt %s --split-input-file --ttatlas-normalize=fold-bias-into-dot=true | FileCheck %s --check-prefixes=CHECK,FLAG-ON

// Pass 2 keeps a dot's bias folded (as the dot's accumulator, which Atlas would load with VMATPUSH.ABF16) when either
// switch is on: the pass flag fold-bias-into-dot, or the module tag ttatlas.fold_bias_into_dot, which compiler.py sets
// from Triton's AtlasOptions.fold_bias_into_dot. Otherwise it unfolds the bias into an add after the dot. Each case
// below is one module-tag setting; the RUN lines run every case with the flag off (as Triton runs it: tag only) and
// with it on. The whole pipeline isn't run here: pass 5 rejects a folded accumulator until it is lowered.
//
//   flag \ tag | absent | false | true
//   off        | unfold | unfold | fold
//   on         | fold   | fold   | fold

// No module tag.
// CHECK-LABEL: tt.func @tag_absent
// FLAG-OFF: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} :
// FLAG-OFF-NEXT: arith.addf %[[DOT]], %arg2
// FLAG-ON: tt.dot %{{.*}}, %{{.*}}, %arg2 :
// FLAG-ON-NOT: arith.addf
// CHECK: tt.return
module {
  tt.func @tag_absent(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %d = tt.dot %a, %b, %bias : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    tt.return %d : tensor<32x32xf32>
  }
}

// -----

// Module tag false.
// CHECK-LABEL: tt.func @tag_false
// FLAG-OFF: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} :
// FLAG-OFF-NEXT: arith.addf %[[DOT]], %arg2
// FLAG-ON: tt.dot %{{.*}}, %{{.*}}, %arg2 :
// FLAG-ON-NOT: arith.addf
// CHECK: tt.return
module attributes {ttatlas.fold_bias_into_dot = false} {
  tt.func @tag_false(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %d = tt.dot %a, %b, %bias : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    tt.return %d : tensor<32x32xf32>
  }
}

// -----

// Module tag true: folded whatever the flag says.
// CHECK-LABEL: tt.func @tag_true
// CHECK: tt.dot %{{.*}}, %{{.*}}, %arg2 :
// CHECK-NOT: arith.addf
// CHECK: tt.return
module attributes {ttatlas.fold_bias_into_dot = true} {
  tt.func @tag_true(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %d = tt.dot %a, %b, %bias : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    tt.return %d : tensor<32x32xf32>
  }
}
