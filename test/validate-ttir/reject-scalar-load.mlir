// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// A scalar tt.load: Atlas loads whole tiles; there is no scalar data path from DRAM in virtual IR.

tt.func @scalar(%p: !tt.ptr<bf16>) {
  // expected-error @below {{atlas: unsupported tt.load: a scalar load; Atlas loads whole tiles}}
  // expected-error @below {{failed to legalize operation 'tt.load' that was explicitly marked illegal}}
  %v = tt.load %p : !tt.ptr<bf16>
  tt.return
}

// -----

// A scalar tt.store: Atlas stores whole tiles too.
tt.func @scalar_store(%p: !tt.ptr<bf16>) {
  %v = arith.constant 1.000000e+00 : bf16
  // expected-error @below {{atlas: unsupported tt.store: a scalar store; Atlas stores whole tiles}}
  // expected-error @below {{failed to legalize operation 'tt.store' that was explicitly marked illegal}}
  tt.store %p, %v : !tt.ptr<bf16>
  tt.return
}
