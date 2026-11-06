// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=1

// ttatlas-verify-contract checks the output contract of one pass (here pass 1: one tt.func, host dtypes on every
// load and store). Each case keeps or breaks that promise.

tt.func @ok(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %v = tt.load %s {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
  tt.store %s, %v {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @no_host_dtype(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  // expected-error @below {{contract of pass 1 broken: missing ttatlas.host_dtype}}
  %v = tt.load %s : tensor<32x32x!tt.ptr<bf16>>
  tt.store %s, %v {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// expected-error @below {{contract of pass 1 broken: expected one tt.func, found 2}}
module {
  tt.func @a() { tt.return }
  tt.func @b() { tt.return }
}
