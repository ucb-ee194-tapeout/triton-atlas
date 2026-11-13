// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Runtime control flow: scf.if, scf.while and cf branches.

// The condition is a constant here, since a runtime argument would be rejected first; scf.if is rejected either way.
tt.func @scf_if() {
  %c = arith.constant true
  // expected-error @below {{atlas: unsupported scf.if: no lowering yet: runtime control flow}}
  // expected-error @below {{failed to legalize operation 'scf.if' that was explicitly marked illegal}}
  scf.if %c {
  }
  tt.return
}

// -----

tt.func @cf_br() {
  // expected-error @below {{atlas: unsupported cf.br: no lowering yet: runtime control flow}}
  // expected-error @below {{failed to legalize operation 'cf.br' that was explicitly marked illegal}}
  cf.br ^next
^next:
  tt.return
}
