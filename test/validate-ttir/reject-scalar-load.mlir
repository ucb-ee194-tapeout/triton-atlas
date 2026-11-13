// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// A scalar tt.load: Atlas loads whole tiles; there is no scalar data path from DRAM in virtual IR.

tt.func @scalar(%p: !tt.ptr<bf16>) {
  // expected-error @below {{atlas: unsupported tt.load: a scalar load; Atlas loads whole tiles}}
  // expected-error @below {{failed to legalize operation 'tt.load' that was explicitly marked illegal}}
  %v = tt.load %p : !tt.ptr<bf16>
  tt.return
}
