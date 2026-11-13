// RUN: triton-atlas-opt %s --ttatlas-validate-ttir | FileCheck %s

// arith.subi is legal: integer subtraction only appears in address math and masks, and the passes that read those
// (pass 4's masks, pass 6's addresses) evaluate it through the shared lane evaluator.

// CHECK-LABEL: tt.func @subi
// CHECK: arith.subi
tt.func @subi(%i: tensor<32x32xi32>) {
  %r = arith.subi %i, %i : tensor<32x32xi32>
  tt.return
}
