// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// scf.for with a runtime trip count: pass 2 unrolls constant-trip loops only.

// The bound is the program id, a runtime value that isn't an argument (a runtime argument is rejected on tt.func).
tt.func @runtime_trip() {
  %n = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-error @below {{atlas: unsupported scf.for: runtime trip count; pass 2 unrolls constant-trip loops only (atlas-mlir accepts only constant DMA addresses)}}
  // expected-error @below {{failed to legalize operation 'scf.for' that was explicitly marked illegal}}
  scf.for %k = %c0 to %n step %c1 : i32 {
  }
  tt.return
}
