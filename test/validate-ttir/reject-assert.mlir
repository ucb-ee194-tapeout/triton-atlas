// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// cf.assert and ub.unreachable: no Atlas meaning.

tt.func @assert() {
  %c = arith.constant true
  // expected-error @below {{atlas: unsupported cf.assert: no Atlas meaning: an assertion or unreachable}}
  // expected-error @below {{failed to legalize operation 'cf.assert' that was explicitly marked illegal}}
  cf.assert %c, "boom"
  tt.return
}
