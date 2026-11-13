// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Parallel loop forms, which Triton's front end doesn't emit.

tt.func @parallel(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{atlas: unsupported scf.parallel: no Atlas meaning: parallel loop forms; Triton's front end doesn't emit them}}
  // expected-error @below {{failed to legalize operation 'scf.parallel' that was explicitly marked illegal}}
  scf.parallel (%iv) = (%c0) to (%c1) step (%c1) {
    scf.reduce
  }
  tt.return
}
